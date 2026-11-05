"""Grasp an item, follow its recorded object trajectory, and release it."""

from __future__ import annotations

import argparse
import importlib.util
import math
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
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
PLACE_HOLD_STEPS = 25
OPEN_STEPS = 50
RETREAT_STEPS = 75
FINAL_HOLD_STEPS = 50
GRASP_STEPS = sum((SETTLE_STEPS, APPROACH_STEPS, APPROACH_HOLD_STEPS, CLOSE_STEPS, CLOSE_HOLD_STEPS))
CONTACT_LOSS_STEPS = 5


def resample_poses(
    poses: np.ndarray, frame_numbers: np.ndarray, tracking_fps: float, playback_speed: float, action_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the complete object path at the IK action rate."""
    frame_times = (frame_numbers - frame_numbers[0]) / (tracking_fps * playback_speed)
    action_times = np.arange(0.0, frame_times[-1], action_dt)
    action_times = np.r_[action_times, frame_times[-1]]
    positions = np.column_stack([np.interp(action_times, frame_times, poses[:, axis, 3]) for axis in range(3)])
    source_quaternions = quat_from_matrix(torch.as_tensor(poses[:, :3, :3], dtype=torch.float64)).numpy()
    left = np.clip(np.searchsorted(frame_times, action_times, side="right") - 1, 0, len(frame_times) - 2)
    fraction = (action_times - frame_times[left]) / (frame_times[left + 1] - frame_times[left])
    first = source_quaternions[left]
    second = source_quaternions[left + 1].copy()
    dot = np.sum(first * second, axis=1)
    second[dot < 0] *= -1
    angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    denominator = np.maximum(np.sin(angle), 1e-8)
    interpolated = (
        np.sin((1 - fraction) * angle)[:, None] * first + np.sin(fraction * angle)[:, None] * second
    ) / denominator[:, None]
    near_same = angle < 1e-4
    near_fraction = fraction[near_same][:, None]
    interpolated[near_same] = (1 - near_fraction) * first[near_same] + near_fraction * second[near_same]
    interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)
    return positions, interpolated


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


def configure_video(cfg: PickPlaceEnvCfg, video_dir: Path, total_steps: int) -> None:
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
            output_filename_prefix="replay",
            video_length=total_steps,
            video_interval=0,
            frame_stride=2,
            fps=25,
        )
    ]


def run_replay(
    env,
    cfg: PickPlaceEnvCfg,
    grasp: dict,
    positions: np.ndarray,
    quaternions: np.ndarray,
    args,
    transition_steps: int,
) -> None:
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
            contact_streak = torch.where(forces >= args.contact_threshold_n, contact_streak + 1, 0)
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

        still_touching = contact_forces(env, torch) >= args.contact_threshold_n
        thumb_ready = stable_contact[:, 0] & still_touching[:, 0]
        opposing_ready = (stable_contact[:, 1] & still_touching[:, 1]) | (stable_contact[:, 2] & still_touching[:, 2])
        if not bool((thumb_ready & opposing_ready).all().item()):
            raise RuntimeError("Grasp failed: thumb and opposing finger contact were not maintained")
        item_shift = float(
            torch.linalg.vector_norm(item.data.root_link_pose_w.torch[:, :3] - object_pose[:, :3]).item()
        )
        print(f"Grasp contacts established; item moved {item_shift:.3f} m since settling.", flush=True)

        # Measure the held relationship once, after the fingers close.
        held_object = item.data.root_link_pose_w.torch.clone()
        held_wrist = robot.data.body_link_pose_w.torch[:, wrist_ids[1]].clone()
        object_to_wrist_pos, object_to_wrist_quat = subtract_frame_transforms(
            held_object[:, :3], held_object[:, 3:7], held_wrist[:, :3], held_wrist[:, 3:7]
        )
        recorded_start_pos = torch.as_tensor(positions[0], device=env.device, dtype=action.dtype).unsqueeze(0)
        recorded_start_quat = torch.as_tensor(quaternions[0], device=env.device, dtype=action.dtype).unsqueeze(0)
        start_pos = held_object[:, :3].clone()
        start_quat = held_object[:, 3:7].clone()
        initial_gap = float(torch.linalg.vector_norm(start_pos - recorded_start_pos, dim=-1).max().item())
        initial_angle = float(torch.rad2deg(quat_error_magnitude(start_quat, recorded_start_quat)).max().item())
        print(f"Transitioning to recorded start: {initial_gap:.3f} m, {initial_angle:.1f} deg.", flush=True)
        if initial_gap > args.max_object_error_m or initial_angle > args.max_object_angle_deg:
            raise RuntimeError("Grasp displaced the item too far from the recording's start pose")
        lost_contact_steps = 0
        max_position_error = 0.0
        max_angle_error = 0.0

        def move_step(stage: str, desired_pos: torch.Tensor, desired_quat: torch.Tensor) -> None:
            nonlocal lost_contact_steps, max_position_error, max_angle_error
            wrist_pos, wrist_quat = combine_frame_transforms(
                desired_pos, desired_quat, object_to_wrist_pos, object_to_wrist_quat
            )
            action[:, 7:10] = wrist_pos - origin
            action[:, 10:14] = wrist_quat
            advance(stage)
            forces = contact_forces(env, torch) >= args.contact_threshold_n
            grip_contact = forces[:, 0] & (forces[:, 1] | forces[:, 2])
            lost_contact_steps = 0 if bool(grip_contact.all().item()) else lost_contact_steps + 1
            if lost_contact_steps >= CONTACT_LOSS_STEPS:
                raise RuntimeError(f"Grasp contact lost during {stage}")
            actual_pose = item.data.root_link_pose_w.torch
            position_error = float(torch.linalg.vector_norm(actual_pose[:, :3] - desired_pos, dim=-1).max().item())
            angle_error = float(torch.rad2deg(quat_error_magnitude(actual_pose[:, 3:7], desired_quat)).max().item())
            max_position_error = max(max_position_error, position_error)
            max_angle_error = max(max_angle_error, angle_error)
            if position_error > args.max_object_error_m or angle_error > args.max_object_angle_deg:
                raise RuntimeError(
                    f"Object departed from target during {stage}: {position_error:.3f} m, {angle_error:.1f} deg"
                )

        for step in range(transition_steps):
            fraction = (step + 1) / transition_steps
            blend = fraction * fraction * (3 - 2 * fraction)
            desired_pos = start_pos + blend * (recorded_start_pos - start_pos)
            desired_quat = quat_slerp(start_quat[0], recorded_start_quat[0], blend).unsqueeze(0)
            move_step("start transition", desired_pos, desired_quat)

        print(f"Following {len(positions)} resampled object poses...", flush=True)
        for index, (position, quaternion) in enumerate(zip(positions, quaternions, strict=True)):
            desired_pos = torch.as_tensor(position, device=env.device, dtype=action.dtype).unsqueeze(0)
            desired_quat = torch.as_tensor(quaternion, device=env.device, dtype=action.dtype).unsqueeze(0)
            move_step(f"tracked pose {index}", desired_pos, desired_quat)

        final_target_pos = torch.as_tensor(positions[-1], device=env.device, dtype=action.dtype).unsqueeze(0)
        final_target_quat = torch.as_tensor(quaternions[-1], device=env.device, dtype=action.dtype).unsqueeze(0)
        for _ in range(PLACE_HOLD_STEPS):
            move_step("placement hold", final_target_pos, final_target_quat)
        held_final_pose = item.data.root_link_pose_w.torch
        held_error = float(torch.linalg.vector_norm(held_final_pose[:, :3] - final_target_pos, dim=-1).max().item())
        held_angle = float(torch.rad2deg(quat_error_magnitude(held_final_pose[:, 3:7], final_target_quat)).max().item())
        if held_error > args.max_place_error_m or held_angle > args.max_place_angle_deg:
            raise RuntimeError(f"Object missed final pose before release: {held_error:.3f} m, {held_angle:.1f} deg")

        print("Opening fingers and withdrawing hand...", flush=True)
        closed_hand = action[:, 14:].clone()
        for step in range(OPEN_STEPS):
            fraction = (step + 1) / OPEN_STEPS
            action[:, 14:] = closed_hand + fraction * (open_hand - closed_hand)
            advance("finger opening")
        retreat_start = action[:, 7:10].clone()
        retreat_delta = torch.tensor([[0.0, -0.10, 0.08]], device=env.device, dtype=action.dtype)
        for step in range(RETREAT_STEPS):
            action[:, 7:10] = retreat_start + ((step + 1) / RETREAT_STEPS) * retreat_delta
            advance("hand retreat")
        for _ in range(FINAL_HOLD_STEPS):
            advance("final settle")

        final_pose = item.data.root_link_pose_w.torch
        place_error = float(torch.linalg.vector_norm(final_pose[:, :3] - final_target_pos, dim=-1).max().item())
        place_angle = float(torch.rad2deg(quat_error_magnitude(final_pose[:, 3:7], final_target_quat)).max().item())
        print(
            f"Maximum path error: {max_position_error:.3f} m, {max_angle_error:.1f} deg; "
            f"final placement error: {place_error:.3f} m, {place_angle:.1f} deg",
            flush=True,
        )
        if place_error > args.max_place_error_m or place_angle > args.max_place_angle_deg:
            raise RuntimeError("Object was not left at the recorded final pose")
        print("Recorded trajectory replay and release completed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--tracking-config", required=True, type=Path)
    parser.add_argument("--grasp-file", type=Path, help="Isaac grasp YAML; defaults to the cracker-box grasps")
    parser.add_argument("--grasp-id", type=int, default=22, help="Grasp ID tested by the contact evaluator")
    parser.add_argument("--tracking-fps", type=float, default=30.0, help="Frame rate of the source recording")
    parser.add_argument("--playback-speed", type=float, default=0.5, help="Trajectory speed relative to the recording")
    parser.add_argument("--transition-duration-s", type=float, default=1.0)
    parser.add_argument("--contact-threshold-n", type=float, default=0.5)
    parser.add_argument("--max-object-error-m", type=float, default=0.05)
    parser.add_argument("--max-object-angle-deg", type=float, default=25.0)
    parser.add_argument("--max-place-error-m", type=float, default=0.05)
    parser.add_argument("--max-place-angle-deg", type=float, default=25.0)
    parser.add_argument("--record-video", action="store_true", help="Save an MP4 of the full replay")
    parser.add_argument("--video-dir", type=Path, help="MP4 directory; implies --record-video")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            args.tracking_fps,
            args.playback_speed,
            args.transition_duration_s,
            args.contact_threshold_n,
            args.max_object_error_m,
            args.max_object_angle_deg,
            args.max_place_error_m,
            args.max_place_angle_deg,
        )
    ):
        parser.error("FPS, speed, transition duration, contact threshold, and error limits must be positive")
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
        action_dt = cfg.sim.dt * cfg.decimation
        positions, quaternions = resample_poses(
            tracked_poses, frame_numbers, args.tracking_fps, args.playback_speed, action_dt
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    transition_steps = max(1, round(args.transition_duration_s / action_dt))
    total_steps = (
        GRASP_STEPS
        + transition_steps
        + len(positions)
        + PLACE_HOLD_STEPS
        + OPEN_STEPS
        + RETREAT_STEPS
        + FINAL_HOLD_STEPS
    )
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = (total_steps + 50) * action_dt
    configure_contact_sensors(cfg)
    video_dir = None
    if args.record_video or args.video_dir is not None:
        video_dir = args.video_dir or Path("outputs/tracking_replays") / (
            f"{args.tracking_config.stem}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        )
        video_dir = video_dir.expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        configure_video(cfg, video_dir, total_steps)

    print(
        f"Item: {item.name}; frames: {len(frame_numbers)}; action targets: {len(positions)}; "
        f"grasp: {args.grasp_id}; playback speed: {args.playback_speed:g}x; grasp file: {grasp_file}",
        flush=True,
    )
    with launch_simulation(cfg, args), tempfile.TemporaryDirectory(prefix="g1_grasp_urdf_") as urdf_dir:
        from isaaclab.envs import ManagerBasedRLEnv

        prepare_ik_urdf(cfg, urdf_dir)
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            if video_dir is not None:
                focus_position = (positions.min(axis=0) + positions.max(axis=0)) / 2
                focus = torch.as_tensor(focus_position, device=env.device, dtype=torch.float32)
                origin = env.scene.env_origins
                env.scene["camera"].set_world_poses_from_view(
                    origin + focus + torch.tensor((0.9, 0.9, 0.65), device=env.device),
                    origin + focus,
                )
            run_replay(env, cfg, grasps[args.grasp_id], positions, quaternions, args, transition_steps)
        finally:
            env.close()
        if video_dir is not None:
            clips = sorted(video_dir.glob("replay_*.mp4"))
            if not clips:
                raise RuntimeError(f"Replay completed but no MP4 was saved in {video_dir}")
            print(f"Replay video: {clips[-1]}", flush=True)


if __name__ == "__main__":
    main()
