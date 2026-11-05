"""Approach and grasp an item at the first pose of a FoundationPose recording."""

from __future__ import annotations

import argparse
import importlib.util
import math
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_contact_grasps import (
    DEFAULT_GRASPS,
    DIGITS,
    configure_contact_sensors,
    configure_item,
    contact_forces,
    load_grasps,
)

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import quat_from_matrix
from isaaclab.visualizers import ImageViewCfg

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg
from g1_real2sim.tracking import load_tracking

SETTLE_STEPS = 25
APPROACH_STEPS = 100
APPROACH_HOLD_STEPS = 50
CLOSE_STEPS = 50
CLOSE_HOLD_STEPS = 25
TOTAL_STEPS = sum((SETTLE_STEPS, APPROACH_STEPS, APPROACH_HOLD_STEPS, CLOSE_STEPS, CLOSE_HOLD_STEPS))


def prepare_ik_urdf(cfg: PickPlaceEnvCfg, directory: str) -> None:
    from isaaclab.controllers.utils import convert_usd_to_urdf

    urdf_path, _ = convert_usd_to_urdf(cfg.scene.robot.spawn.usd_path, directory)
    urdf_file = Path(urdf_path)
    tree = ET.parse(urdf_file)
    for mesh in tree.iter("mesh"):
        if filename := mesh.get("filename"):
            mesh.set("filename", str((urdf_file.parent / filename).resolve()))
    tree.write(urdf_file)
    controller = cfg.actions.upper_body_ik.controller
    controller.urdf_path = urdf_path
    controller.mesh_path = None
    controller.usd_path = None


def configure_video(cfg: PickPlaceEnvCfg, video_dir: Path) -> None:
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
    )
    cfg.video_recorders = [
        VideoRecorderCfg(
            source=ImageViewCfg(source="camera", envs=(0,)),
            output_dir=str(video_dir),
            output_filename_prefix="grasp",
            video_length=TOTAL_STEPS,
            video_interval=0,
            frame_stride=2,
            fps=25,
        )
    ]


def run_grasp(env, cfg: PickPlaceEnvCfg, grasp: dict, contact_threshold_n: float) -> None:
    from isaaclab.utils.math import (
        combine_frame_transforms,
        quat_error_magnitude,
        quat_from_euler_xyz,
        quat_slerp,
        subtract_frame_transforms,
    )

    robot = env.scene["robot"]
    item = env.scene["object"]
    origin = env.scene.env_origins
    wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    hand_ids, hand_names = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)
    action = torch.zeros_like(env.action_manager.action)
    wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
    wrist_poses[..., :3] -= origin[:, None, :]
    action[:, :14] = wrist_poses.flatten(start_dim=1)
    action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]
    open_hand = action[:, 14:].clone()

    def advance(stage: str) -> None:
        _, _, terminated, truncated, _ = env.step(action)
        if bool(torch.any(terminated | truncated)):
            raise RuntimeError(f"Episode reset during {stage}")

    with torch.inference_mode():
        for _ in range(SETTLE_STEPS):
            advance("item settle")
        object_pose = item.data.root_link_pose_w.torch.clone()

        grasp_position = torch.tensor([grasp["position"]], device=env.device, dtype=action.dtype)
        grasp_quaternion = torch.tensor([grasp["quaternion_xyzw"]], device=env.device, dtype=action.dtype)
        grasp_pos_w, grasp_quat_w = combine_frame_transforms(
            object_pose[:, :3], object_pose[:, 3:7], grasp_position, grasp_quaternion
        )
        root_to_palm_pos = torch.tensor([[0.07, 0.0, 0.02]], device=env.device, dtype=action.dtype)
        root_to_palm_quat = quat_from_euler_xyz(
            torch.tensor([1.5708], device=env.device, dtype=action.dtype),
            torch.tensor([0.0], device=env.device, dtype=action.dtype),
            torch.tensor([3.14159], device=env.device, dtype=action.dtype),
        )
        palm_pos_w, palm_quat_w = combine_frame_transforms(
            grasp_pos_w, grasp_quat_w, root_to_palm_pos, root_to_palm_quat
        )
        links = robot.data.body_link_pose_w.torch
        wrist_pose = links[:, wrist_ids[1]]
        palm_pose = links[:, robot.body_names.index("right_hand_palm_link")]
        palm_to_wrist_pos, palm_to_wrist_quat = subtract_frame_transforms(
            palm_pose[:, :3], palm_pose[:, 3:7], wrist_pose[:, :3], wrist_pose[:, 3:7]
        )
        target_pos_w, target_quat = combine_frame_transforms(
            palm_pos_w, palm_quat_w, palm_to_wrist_pos, palm_to_wrist_quat
        )

        print("Approaching grasp...", flush=True)
        start_pos = action[:, 7:10].clone()
        start_quat = action[:, 10:14].clone()
        for step in range(APPROACH_STEPS):
            fraction = (step + 1) / APPROACH_STEPS
            action[:, 7:10] = start_pos + fraction * (target_pos_w - origin - start_pos)
            action[:, 10:14] = quat_slerp(start_quat[0], target_quat[0], fraction).unsqueeze(0)
            advance("grasp approach")
        for _ in range(APPROACH_HOLD_STEPS):
            advance("grasp hold")
        actual_wrist = robot.data.body_link_pose_w.torch[:, wrist_ids[1]]
        wrist_error = float(torch.linalg.vector_norm(actual_wrist[:, :3] - target_pos_w, dim=-1).max().item())
        angle_error = float(torch.rad2deg(quat_error_magnitude(actual_wrist[:, 3:7], target_quat)).max().item())
        if wrist_error > 0.04 or angle_error > 20:
            raise RuntimeError(f"Grasp wrist target was not reached: {wrist_error:.3f} m, {angle_error:.1f} deg")

        print("Closing fingers on the object...", flush=True)
        limits = robot.data.soft_joint_pos_limits.torch[:, hand_ids, :]
        hand_delta = torch.zeros_like(open_hand)
        digit_joint_ids = {digit: [] for digit in DIGITS}
        for hand_index, name in enumerate(hand_names):
            if name.startswith("right_hand_") and name != "right_hand_thumb_0_joint":
                digit = name.split("_")[2]
                limit_index = 0 if digit == "thumb" else 1
                hand_delta[:, hand_index] = limits[:, hand_index, limit_index] - open_hand[:, hand_index]
                if not bool(torch.all(torch.abs(hand_delta[:, hand_index]) > 0.05)):
                    raise ValueError(f"No closing range for {name}")
                digit_joint_ids[digit].append(hand_index)
        if any(not ids for ids in digit_joint_ids.values()):
            raise RuntimeError(f"Missing right-hand finger joints: {digit_joint_ids}")

        digit_action_ids = {digit: 14 + torch.tensor(ids, device=env.device) for digit, ids in digit_joint_ids.items()}
        digit_fraction = torch.zeros((1, 3), device=env.device)
        contact_streak = torch.zeros((1, 3), device=env.device, dtype=torch.int32)
        stable_contact = torch.zeros((1, 3), device=env.device, dtype=torch.bool)
        for _ in range(CLOSE_STEPS):
            digit_fraction = torch.where(
                stable_contact, digit_fraction, torch.clamp(digit_fraction + 0.9 / CLOSE_STEPS, max=0.9)
            )
            for digit_index, digit in enumerate(DIGITS):
                ids = digit_joint_ids[digit]
                action[:, digit_action_ids[digit]] = (
                    open_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
                )
            advance("finger closure")
            forces = contact_forces(env, torch)
            contact_streak = torch.where(forces >= contact_threshold_n, contact_streak + 1, 0)
            newly_stable = ~stable_contact & (contact_streak >= 3)
            digit_fraction = torch.where(newly_stable, torch.clamp(digit_fraction + 0.02, max=0.9), digit_fraction)
            stable_contact |= newly_stable
        for digit_index, digit in enumerate(DIGITS):
            ids = digit_joint_ids[digit]
            action[:, digit_action_ids[digit]] = (
                open_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
            )
        for _ in range(CLOSE_HOLD_STEPS):
            advance("grasp settle")

        still_touching = contact_forces(env, torch) >= contact_threshold_n
        thumb_ready = stable_contact[:, 0] & still_touching[:, 0]
        opposing_ready = (stable_contact[:, 1] & still_touching[:, 1]) | (stable_contact[:, 2] & still_touching[:, 2])
        if not bool((thumb_ready & opposing_ready).all().item()):
            raise RuntimeError("Grasp failed: thumb and opposing finger contact were not maintained")
        item_shift = float(
            torch.linalg.vector_norm(item.data.root_link_pose_w.torch[:, :3] - object_pose[:, :3]).item()
        )
        print(f"Grasp contacts established; item moved {item_shift:.3f} m since settling.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--tracking-config", required=True, type=Path)
    parser.add_argument("--grasp-file", type=Path, help="Isaac grasp YAML; defaults to the cracker-box grasps")
    parser.add_argument("--grasp-id", type=int, default=22, help="Grasp ID tested by the contact evaluator")
    parser.add_argument("--contact-threshold-n", type=float, default=0.5)
    parser.add_argument("--record-video", action="store_true", help="Save an MP4 of the grasp attempt")
    parser.add_argument("--video-dir", type=Path, help="MP4 directory; implies --record-video")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if not math.isfinite(args.contact_threshold_n) or args.contact_threshold_n <= 0:
        parser.error("--contact-threshold-n must be positive")
    if (args.record_video or args.video_dir is not None) and importlib.util.find_spec("moviepy") is None:
        parser.error("Video recording requires: uv run --extra isaacsim --extra video ...")

    try:
        item_path, frame_numbers, tracked_poses = load_tracking(args.tracking_config)
        first_position = tuple(float(value) for value in tracked_poses[0, :3, 3])
        first_quaternion = tuple(
            float(value)
            for value in quat_from_matrix(torch.as_tensor(tracked_poses[0, :3, :3], dtype=torch.float64)).numpy()
        )
        cfg = PickPlaceEnvCfg()
        item = configure_item(
            cfg,
            argparse.Namespace(item_config=item_path, item_position=first_position, item_rotation=first_quaternion),
        )
        grasp_file = args.grasp_file or item.grasp_file or (DEFAULT_GRASPS if item.name == "cracker_box" else None)
        if grasp_file is None:
            raise ValueError(f"Provide --grasp-file for {item.name}")
        grasp_file = grasp_file.expanduser().resolve(strict=True)
        grasps = load_grasps(grasp_file)
        if args.grasp_id not in grasps:
            raise ValueError(f"Grasp {args.grasp_id} is missing from {grasp_file}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = (TOTAL_STEPS + 50) * cfg.sim.dt * cfg.decimation
    configure_contact_sensors(cfg)
    video_dir = None
    if args.record_video or args.video_dir is not None:
        video_dir = args.video_dir or Path("outputs/tracked_grasps") / (
            f"{args.tracking_config.stem}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        )
        video_dir = video_dir.expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        configure_video(cfg, video_dir)

    print(
        f"Item: {item.name}; first frame: {int(frame_numbers[0])}; grasp: {args.grasp_id}; grasp file: {grasp_file}",
        flush=True,
    )
    with launch_simulation(cfg, args), tempfile.TemporaryDirectory(prefix="g1_grasp_urdf_") as urdf_dir:
        from isaaclab.envs import ManagerBasedRLEnv

        prepare_ik_urdf(cfg, urdf_dir)
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            if video_dir is not None:
                focus = torch.as_tensor(first_position, device=env.device, dtype=torch.float32)
                origin = env.scene.env_origins
                env.scene["camera"].set_world_poses_from_view(
                    origin + focus + torch.tensor((0.9, 0.9, 0.65), device=env.device),
                    origin + focus,
                )
            run_grasp(env, cfg, grasps[args.grasp_id], args.contact_threshold_n)
        finally:
            env.close()
        if video_dir is not None:
            clips = sorted(video_dir.glob("grasp_*.mp4"))
            if not clips:
                raise RuntimeError(f"Grasp completed but no MP4 was saved in {video_dir}")
            print(f"Grasp video: {clips[-1]}", flush=True)


if __name__ == "__main__":
    main()
