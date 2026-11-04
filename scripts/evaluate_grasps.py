"""Try all 100 cracker-box grasps in parallel with physical contact and report lifts."""

import argparse
import csv
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from isaaclab.app import add_launcher_args, launch_simulation

from g1_real2sim.assets import G1_REAL2SIM_ASSETS_DIR
from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def load_grasps() -> list[dict]:
    grasp_file = G1_REAL2SIM_ASSETS_DIR / "grasps/cracker_box_grasps.yml"
    with grasp_file.open() as file:
        data = yaml.safe_load(file)["grasps"]
    expected = {f"grasp_{index}" for index in range(100)}
    if set(data) != expected:
        raise ValueError(f"Expected grasp_0 through grasp_99 in {grasp_file}")
    return [data[f"grasp_{index}"] for index in range(100)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--output-file", type=Path, default=Path("outputs/grasp_evaluation/results.csv"))
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    grasps = load_grasps()
    cfg = PickPlaceEnvCfg()
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = len(grasps)
    cfg.episode_length_s = 30.0

    with launch_simulation(cfg, args), tempfile.TemporaryDirectory(prefix="g1_urdf_") as urdf_dir:
        import torch

        from isaaclab.controllers.utils import convert_usd_to_urdf
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.math import (
            combine_frame_transforms,
            quat_error_magnitude,
            quat_from_euler_xyz,
            subtract_frame_transforms,
        )

        # Pink uses an exported model of the exact USD spawned in this scene.
        urdf_path, _ = convert_usd_to_urdf(cfg.scene.robot.spawn.usd_path, urdf_dir)
        urdf_file = Path(urdf_path)
        urdf_tree = ET.parse(urdf_file)
        for mesh in urdf_tree.iter("mesh"):
            if filename := mesh.get("filename"):
                mesh.set("filename", str((urdf_file.parent / filename).resolve()))
        urdf_tree.write(urdf_file)
        controller = cfg.actions.upper_body_ik.controller
        controller.urdf_path = urdf_path
        controller.mesh_path = None
        controller.usd_path = None

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            box = env.scene["object"]
            origins = env.scene.env_origins
            wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
            hand_ids, hand_names = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)
            wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
            wrist_poses[..., :3] -= origins[:, None, :]
            action = torch.zeros_like(env.action_manager.action)
            action[:, :14] = wrist_poses.flatten(start_dim=1)
            action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]

            # Let the box settle on the table before fixing its grasp target and baseline height.
            with torch.inference_mode():
                for _ in range(25):
                    env.step(action)
            object_pose = box.data.root_link_pose_w.torch.clone()
            initial_z = object_pose[:, 2].clone()
            grasp_pos = torch.tensor(
                [grasp["position"] for grasp in grasps], device=env.device, dtype=object_pose.dtype
            )
            grasp_quat = torch.tensor(
                [[*grasp["orientation"]["xyz"], grasp["orientation"]["w"]] for grasp in grasps],
                device=env.device,
                dtype=object_pose.dtype,
            )
            grasp_pos_w, grasp_quat_w = combine_frame_transforms(
                object_pose[:, :3], object_pose[:, 3:7], grasp_pos, grasp_quat
            )

            # Same GraspGenX gripper-root -> palm calibration as visualize_grasps.py.
            root_to_palm_pos = torch.tensor([[0.07, 0.0, 0.02]], device=env.device, dtype=object_pose.dtype).expand(
                len(grasps), -1
            )
            root_to_palm_quat = quat_from_euler_xyz(
                torch.tensor([1.5708], device=env.device),
                torch.tensor([0.0], device=env.device),
                torch.tensor([3.14159], device=env.device),
            ).expand(len(grasps), -1)
            palm_pos_w, palm_quat_w = combine_frame_transforms(
                grasp_pos_w, grasp_quat_w, root_to_palm_pos, root_to_palm_quat
            )
            poses = robot.data.body_link_pose_w.torch
            wrist_pose = poses[:, wrist_ids[1]]
            palm_pose = poses[:, robot.body_names.index("right_hand_palm_link")]
            palm_to_wrist_pos, palm_to_wrist_quat = subtract_frame_transforms(
                palm_pose[:, :3], palm_pose[:, 3:7], wrist_pose[:, :3], wrist_pose[:, 3:7]
            )
            target_pos_w, target_quat = combine_frame_transforms(
                palm_pos_w, palm_quat_w, palm_to_wrist_pos, palm_to_wrist_quat
            )
            target_pos = target_pos_w - origins

            start_pos = action[:, 7:10].clone()
            start_quat = action[:, 10:14].clone()
            dot = (start_quat * target_quat).sum(dim=-1, keepdim=True)
            end_quat = torch.where(dot < 0, -target_quat, target_quat)
            angle = torch.acos(dot.abs().clamp(max=1.0))
            sin_angle = torch.sin(angle)

            # Keep the left hand and thumb yaw at reset. The Dex3 thumb flexes toward
            # its lower limits; the index and middle fingers flex toward upper limits.
            close_action = action[:, 14:].clone()
            limits = robot.data.soft_joint_pos_limits.torch[:, hand_ids, :]
            for hand_index, name in enumerate(hand_names):
                if name.startswith("right_hand_") and name != "right_hand_thumb_0_joint":
                    opened = action[:, 14 + hand_index]
                    limit_index = 0 if name.startswith("right_hand_thumb_") else 1
                    closed_limit = limits[:, hand_index, limit_index]
                    if not bool(torch.all(torch.abs(closed_limit - opened) > 0.05)):
                        raise ValueError(f"No closing range for {name}")
                    close_action[:, hand_index] = opened + 0.9 * (closed_limit - opened)

            def step(stage: str, count: int, update_lift: bool = False) -> None:
                nonlocal max_rise, hold_streak
                with torch.inference_mode():
                    for _ in range(count):
                        _, _, terminated, truncated, _ = env.step(action)
                        if bool(torch.any(terminated | truncated)):
                            raise RuntimeError(f"An environment reset during {stage}; increase episode_length_s")
                        if update_lift:
                            rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
                            max_rise = torch.maximum(max_rise, rise)
                            hold_streak = torch.where(rise >= 0.10, hold_streak + 1, 0)
                print(f"Finished {stage}", flush=True)

            max_rise = torch.zeros(len(grasps), device=env.device)
            hold_streak = torch.zeros(len(grasps), device=env.device, dtype=torch.int32)
            with torch.inference_mode():
                for index in range(100):
                    fraction = (index + 1) / 100
                    action[:, 7:10] = start_pos + fraction * (target_pos - start_pos)
                    slerp_quat = (
                        torch.sin((1 - fraction) * angle) * start_quat + torch.sin(fraction * angle) * end_quat
                    ) / sin_angle.clamp_min(1e-6)
                    lerp_quat = torch.nn.functional.normalize((1 - fraction) * start_quat + fraction * end_quat, dim=-1)
                    action[:, 10:14] = torch.where(sin_angle > 1e-6, slerp_quat, lerp_quat)
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool(torch.any(terminated | truncated)):
                        raise RuntimeError("An environment reset during approach")
            step("grasp pose hold", 50)
            actual_pose = robot.data.body_link_pose_w.torch[:, wrist_ids[1]]
            position_errors = torch.linalg.norm(actual_pose[:, :3] - target_pos_w, dim=-1).clone()
            rotation_errors = torch.rad2deg(quat_error_magnitude(actual_pose[:, 3:7], target_quat)).clone()

            start_hand = action[:, 14:].clone()
            with torch.inference_mode():
                for index in range(50):
                    fraction = (index + 1) / 50
                    action[:, 14:] = start_hand + fraction * (close_action - start_hand)
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool(torch.any(terminated | truncated)):
                        raise RuntimeError("An environment reset during hand closure")
            step("grasp settle", 25)

            lift_start = action[:, 7:10].clone()
            with torch.inference_mode():
                for index in range(75):
                    action[:, 7:10] = lift_start + (index + 1) / 75 * torch.tensor(
                        [0.0, 0.0, 0.15], device=env.device, dtype=action.dtype
                    )
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool(torch.any(terminated | truncated)):
                        raise RuntimeError("An environment reset during lift")
                    rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
                    max_rise = torch.maximum(max_rise, rise)
            step("lift hold", 50, update_lift=True)

            final_rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
            success = (hold_streak >= math.ceil(0.5 / (cfg.sim.dt * cfg.decimation))) & (final_rise >= 0.10)
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            with args.output_file.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "grasp_id",
                        "confidence",
                        "success",
                        "failure_reason",
                        "max_box_rise_m",
                        "final_box_rise_m",
                        "wrist_position_error_m",
                        "wrist_rotation_error_deg",
                    ],
                )
                writer.writeheader()
                for index, grasp in enumerate(grasps):
                    succeeded = bool(success[index].item())
                    if succeeded:
                        reason = ""
                    elif position_errors[index] > 0.05 or rotation_errors[index] > 30.0:
                        reason = "wrist_pose_not_reached"
                    else:
                        reason = "box_not_held_above_10cm"
                    writer.writerow(
                        {
                            "grasp_id": f"grasp_{index}",
                            "confidence": grasp["confidence"],
                            "success": succeeded,
                            "failure_reason": reason,
                            "max_box_rise_m": float(max_rise[index].item()),
                            "final_box_rise_m": float(final_rise[index].item()),
                            "wrist_position_error_m": float(position_errors[index].item()),
                            "wrist_rotation_error_deg": float(rotation_errors[index].item()),
                        }
                    )
            successful_ids = [f"grasp_{index}" for index in range(len(grasps)) if bool(success[index].item())]
            print(f"Successful grasps ({len(successful_ids)}/100): {', '.join(successful_ids) or 'none'}")
            print(f"Results saved to {args.output_file}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
