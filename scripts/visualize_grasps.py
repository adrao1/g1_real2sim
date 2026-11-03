"""Move the G1 right wrist to grasp_0 and save an image."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, UsdPhysicsCollisionCfg

from g1_real2sim.assets import G1_REAL2SIM_ASSETS_DIR
from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    grasp_file = G1_REAL2SIM_ASSETS_DIR / "grasps/cracker_box_grasps.yml"
    with grasp_file.open() as file:
        grasp = yaml.safe_load(file)["grasps"]["grasp_0"]

    cfg = PickPlaceEnvCfg()
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = 1
    cfg.scene.object.spawn.collision_props = UsdPhysicsCollisionCfg(collision_enabled=False)
    cfg.scene.object.spawn.rigid_props.kinematic_enabled = True
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
    )
    with launch_simulation(cfg, args), tempfile.TemporaryDirectory(prefix="g1_urdf_") as urdf_dir:
        import torch

        from isaaclab.controllers.utils import convert_usd_to_urdf
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors.camera.utils import save_images_to_file
        from isaaclab.utils.math import (
            combine_frame_transforms,
            quat_error_magnitude,
            quat_from_euler_xyz,
            quat_slerp,
            subtract_frame_transforms,
        )

        print("Converting G1 USD to URDF for IK...", flush=True)
        urdf_path, _ = convert_usd_to_urdf(cfg.scene.robot.spawn.usd_path, urdf_dir)
        urdf_file = Path(urdf_path)
        urdf_tree = ET.parse(urdf_file)
        for mesh in urdf_tree.iter("mesh"):
            if filename := mesh.get("filename"):
                mesh.set("filename", str((urdf_file.parent / filename).resolve()))
        urdf_tree.write(urdf_file)
        cfg.actions.upper_body_ik.controller.urdf_path = urdf_path
        cfg.actions.upper_body_ik.controller.mesh_path = None
        cfg.actions.upper_body_ik.controller.usd_path = None

        print(f"Creating {cfg.scene.num_envs} environments...", flush=True)
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            print("Environments created; resetting...", flush=True)
            env.reset()
            robot = env.scene["robot"]
            camera = env.scene["camera"]
            origins = env.scene.env_origins
            camera.set_world_poses_from_view(
                origins + torch.tensor((1.2, 1.2, 1.5), device=env.device),
                origins + torch.tensor((0.0, 0.4, 0.85), device=env.device),
            )
            wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
            wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
            wrist_poses[..., :3] -= origins[:, None, :]
            hand_ids, _ = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)

            action = torch.zeros_like(env.action_manager.action)
            action[:, :14] = wrist_poses.flatten(start_dim=1)
            action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]

            object_pose = env.scene["object"].data.root_link_pose_w.torch
            grasp_pos = torch.tensor(grasp["position"], device=env.device, dtype=object_pose.dtype).unsqueeze(0)
            orientation = grasp["orientation"]
            grasp_quat = torch.tensor(
                [*orientation["xyz"], orientation["w"]], device=env.device, dtype=object_pose.dtype
            ).unsqueeze(0)
            grasp_pos_w, grasp_quat_w = combine_frame_transforms(
                object_pose[:, :3], object_pose[:, 3:7], grasp_pos, grasp_quat
            )

            # GraspGenX unitree_g1/gripper.urdf: world_joint -> right_palm_link.
            root_to_palm_pos = torch.tensor([[0.07, 0.0, 0.02]], device=env.device, dtype=object_pose.dtype)
            root_to_palm_quat = quat_from_euler_xyz(
                torch.tensor([1.5708], device=env.device),
                torch.tensor([0.0], device=env.device),
                torch.tensor([3.14159], device=env.device),
            )
            palm_pos_w, palm_quat_w = combine_frame_transforms(
                grasp_pos_w, grasp_quat_w, root_to_palm_pos, root_to_palm_quat
            )

            # Assume the GraspGenX palm frame matches the simulator palm frame.
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
            start_quat = action[0, 10:14].clone()
            print(f"grasp_0 right wrist target position: {target_pos[0].tolist()}")
            print(f"grasp_0 right wrist target rotation (xyzw): {target_quat[0].tolist()}")

            with torch.inference_mode():
                for step in range(100):
                    fraction = (step + 1) / 100
                    action[:, 7:10] = start_pos + fraction * (target_pos - start_pos)
                    action[0, 10:14] = quat_slerp(start_quat, target_quat[0], fraction)
                    env.step(action)
                    if (step + 1) % 25 == 0:
                        print(f"Move step {step + 1}/100", flush=True)
                print("Holding final pose for 100 steps...", flush=True)
                for _ in range(100):
                    env.step(action)

            actual_pose = robot.data.body_link_pose_w.torch[0, wrist_ids[1]]
            position_error = torch.linalg.norm(actual_pose[:3] - target_pos_w[0]).item()
            rotation_error = torch.rad2deg(quat_error_magnitude(actual_pose[3:7], target_quat[0])).item()
            print(f"grasp_0 wrist error: {position_error:.4f} m, {rotation_error:.1f} deg")
            print("Capturing camera images...", flush=True)
            images = camera.data.output["rgb"].torch
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image_path = args.output_dir / "grasp_0.png"
            save_images_to_file(images.float() / 255.0, str(image_path))
            print(f"Saved {image_path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
