"""Shared physical pick-and-lift runner for finger-closure experiments."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

from isaaclab.app import launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.visualizers import ImageViewCfg

from g1_real2sim.assets import G1_REAL2SIM_ASSETS_DIR
from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg

SETTLE_STEPS = 25
APPROACH_STEPS = 100
GRASP_HOLD_STEPS = 50
CLOSE_STEPS = 50
CLOSE_HOLD_STEPS = 25
LIFT_STEPS = 75
LIFT_HOLD_STEPS = 50
TOTAL_STEPS = sum(
    (SETTLE_STEPS, APPROACH_STEPS, GRASP_HOLD_STEPS, CLOSE_STEPS, CLOSE_HOLD_STEPS, LIFT_STEPS, LIFT_HOLD_STEPS)
)
DIGITS = ("thumb", "index", "middle")
CONTACT_LINKS = {
    "thumb": ("right_hand_thumb_2_link",),
    "index": ("right_hand_index_0_link", "right_hand_index_1_link"),
    "middle": ("right_hand_middle_0_link", "right_hand_middle_1_link"),
}


@dataclass(frozen=True)
class Trial:
    grasp_id: int
    percent: int | None = None
    repeat: int = 1
    stage: str = "trial"

    @property
    def label(self) -> str:
        suffix = f"_p{self.percent:02d}" if self.percent is not None else ""
        return f"grasp_{self.grasp_id}{suffix}_{self.stage}_{self.repeat}"


def select_grasps(grasp_ids: list[int] | None, successful_from: Path | None) -> list[int]:
    """Use explicit IDs, a result CSV, or the existing 100-grasp result by default."""
    if grasp_ids is None:
        source = successful_from or Path("outputs/grasp_evaluation/results.csv")
        with source.open(newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"grasp_id", "success"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected grasp_id and success columns in {source}")
            grasp_ids = [
                int(row["grasp_id"].removeprefix("grasp_"))
                for row in reader
                if row["success"].strip().lower() == "true"
            ]
    if not grasp_ids or any(index not in range(100) for index in grasp_ids):
        raise ValueError("Select one or more grasp IDs from 0 to 99")
    if len(set(grasp_ids)) != len(grasp_ids):
        raise ValueError("Each grasp ID may be selected only once")
    return grasp_ids


def load_grasps() -> list[dict]:
    grasp_file = G1_REAL2SIM_ASSETS_DIR / "grasps/cracker_box_grasps.yml"
    with grasp_file.open() as file:
        data = yaml.safe_load(file)["grasps"]
    if set(data) != {f"grasp_{index}" for index in range(100)}:
        raise ValueError(f"Expected grasp_0 through grasp_99 in {grasp_file}")
    return [data[f"grasp_{index}"] for index in range(100)]


def add_selection_args(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--grasps", type=int, nargs="+", metavar="ID", help="Grasp IDs to try (0-99)")
    group.add_argument("--successful-from", type=Path, help="Use successful IDs from a results CSV")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record contact, direct fixed, or fixed-search confirmation trials as separate MP4s",
    )
    parser.add_argument("--video-dir", type=Path, default=None)


def _configure_contact_sensors(cfg: PickPlaceEnvCfg) -> None:
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.object.spawn.activate_contact_sensors = True
    for links in CONTACT_LINKS.values():
        for link in links:
            setattr(
                cfg.scene,
                f"contact_{link}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    update_period=0.0,
                ),
            )


def _contact_forces(env, torch):
    """Return [environment, digit] box-only normal-force magnitudes in newtons."""
    digit_forces = []
    for digit in DIGITS:
        link_forces = []
        for link in CONTACT_LINKS[digit]:
            sensor = env.scene[f"contact_{link}"]
            matrix = sensor.data.normal_force_matrix_w
            if matrix is None:
                raise RuntimeError(f"No filtered force data for {link}")
            values = matrix.torch
            if values.ndim != 4 or values.shape[1:] != (1, 1, 3):
                raise RuntimeError(f"Expected one finger body and one box filter for {link}; got {tuple(values.shape)}")
            link_forces.append(torch.linalg.vector_norm(values[:, 0, 0, :], dim=-1))
        digit_forces.append(torch.stack(link_forces, dim=1).amax(dim=1))
    return torch.stack(digit_forces, dim=1)


def run_batch(
    trials: list[Trial],
    method: str,
    launcher_args,
    result_file: Path,
    video_dir: Path | None = None,
    contact_threshold_n: float = 0.5,
) -> list[dict]:
    """Run one fresh physical pick per environment and return per-trial measurements."""
    if not trials or len(trials) > 100:
        raise ValueError("Run between 1 and 100 trials per batch")
    if method not in ("fixed", "contact"):
        raise ValueError(f"Unknown closure method: {method}")
    if method == "fixed" and any(trial.percent is None or trial.percent not in range(1, 91) for trial in trials):
        raise ValueError("Fixed trials require closure percentages from 1 to 90")
    if video_dir is not None:
        try:
            import moviepy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Videos require: uv run --extra isaacsim --extra video ...") from exc

    grasps = load_grasps()
    cfg = PickPlaceEnvCfg()
    cfg.sim.device = launcher_args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = len(trials)
    cfg.episode_length_s = 30.0
    if method == "contact":
        _configure_contact_sensors(cfg)
    if video_dir is not None:
        cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0,
            width=480,
            height=360,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
        )
        cfg.video_recorders = [
            VideoRecorderCfg(
                source=ImageViewCfg(source="camera", envs=(index,)),
                output_dir=str(video_dir),
                output_filename_prefix=f"{method}_{trial.label}",
                video_length=TOTAL_STEPS,
                frame_stride=2,
                fps=25,
            )
            for index, trial in enumerate(trials)
        ]

    with launch_simulation(cfg, launcher_args), tempfile.TemporaryDirectory(prefix="g1_urdf_") as urdf_dir:
        import torch

        from isaaclab.controllers.utils import convert_usd_to_urdf
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.math import (
            combine_frame_transforms,
            quat_error_magnitude,
            quat_from_euler_xyz,
            subtract_frame_transforms,
        )

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

        print(f"Starting {len(trials)} fresh {method} trials...", flush=True)
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            box = env.scene["object"]
            origins = env.scene.env_origins
            if video_dir is not None:
                env.scene["camera"].set_world_poses_from_view(
                    origins + torch.tensor((1.2, 1.2, 1.5), device=env.device),
                    origins + torch.tensor((0.0, 0.4, 0.85), device=env.device),
                )
            wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
            hand_ids, hand_names = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)
            wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
            wrist_poses[..., :3] -= origins[:, None, :]
            action = torch.zeros_like(env.action_manager.action)
            action[:, :14] = wrist_poses.flatten(start_dim=1)
            action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]

            def advance(stage: str, update_lift: bool = False) -> None:
                nonlocal max_rise, hold_streak
                _, _, terminated, truncated, _ = env.step(action)
                if bool(torch.any(terminated | truncated)):
                    raise RuntimeError(f"An environment reset during {stage}")
                if update_lift:
                    rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
                    max_rise = torch.maximum(max_rise, rise)
                    hold_streak = torch.where(rise >= 0.10, hold_streak + 1, 0)

            max_rise = torch.zeros(len(trials), device=env.device)
            hold_streak = torch.zeros(len(trials), device=env.device, dtype=torch.int32)
            with torch.inference_mode():
                for _ in range(SETTLE_STEPS):
                    advance("box settle")
                object_pose = box.data.root_link_pose_w.torch.clone()
                initial_z = object_pose[:, 2].clone()
                grasp_batch = [grasps[trial.grasp_id] for trial in trials]
                grasp_pos = torch.tensor(
                    [grasp["position"] for grasp in grasp_batch], device=env.device, dtype=object_pose.dtype
                )
                grasp_quat = torch.tensor(
                    [[*grasp["orientation"]["xyz"], grasp["orientation"]["w"]] for grasp in grasp_batch],
                    device=env.device,
                    dtype=object_pose.dtype,
                )
                grasp_pos_w, grasp_quat_w = combine_frame_transforms(
                    object_pose[:, :3], object_pose[:, 3:7], grasp_pos, grasp_quat
                )
                root_to_palm_pos = torch.tensor([[0.07, 0.0, 0.02]], device=env.device, dtype=object_pose.dtype).expand(
                    len(trials), -1
                )
                root_to_palm_quat = quat_from_euler_xyz(
                    torch.tensor([1.5708], device=env.device),
                    torch.tensor([0.0], device=env.device),
                    torch.tensor([3.14159], device=env.device),
                ).expand(len(trials), -1)
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

                for index in range(APPROACH_STEPS):
                    fraction = (index + 1) / APPROACH_STEPS
                    action[:, 7:10] = start_pos + fraction * (target_pos - start_pos)
                    slerp = (
                        torch.sin((1 - fraction) * angle) * start_quat + torch.sin(fraction * angle) * end_quat
                    ) / sin_angle.clamp_min(1e-6)
                    lerp = torch.nn.functional.normalize((1 - fraction) * start_quat + fraction * end_quat, dim=-1)
                    action[:, 10:14] = torch.where(sin_angle > 1e-6, slerp, lerp)
                    advance("approach")
                for _ in range(GRASP_HOLD_STEPS):
                    advance("grasp hold")
                actual_wrist = robot.data.body_link_pose_w.torch[:, wrist_ids[1]]
                position_error = torch.linalg.vector_norm(actual_wrist[:, :3] - target_pos_w, dim=-1).clone()
                rotation_error = torch.rad2deg(quat_error_magnitude(actual_wrist[:, 3:7], target_quat)).clone()
                closure_box_xy = box.data.root_link_pose_w.torch[:, :2].clone()
                box_slide = torch.zeros(len(trials), device=env.device)

                start_hand = action[:, 14:].clone()
                limits = robot.data.soft_joint_pos_limits.torch[:, hand_ids, :]
                hand_delta = torch.zeros_like(start_hand)
                digit_joint_ids = {digit: [] for digit in DIGITS}
                for hand_index, name in enumerate(hand_names):
                    if name.startswith("right_hand_") and name != "right_hand_thumb_0_joint":
                        digit = name.split("_")[2]
                        limit_index = 0 if digit == "thumb" else 1
                        hand_delta[:, hand_index] = limits[:, hand_index, limit_index] - start_hand[:, hand_index]
                        if not bool(torch.all(torch.abs(hand_delta[:, hand_index]) > 0.05)):
                            raise ValueError(f"No closing range for {name}")
                        digit_joint_ids[digit].append(hand_index)
                if any(not ids for ids in digit_joint_ids.values()):
                    raise RuntimeError(f"Missing right-hand joints: {digit_joint_ids}")

                stable_contact = torch.zeros((len(trials), 3), device=env.device, dtype=torch.bool)
                peak_force = torch.zeros((len(trials), 3), device=env.device)
                force_at_stop = torch.zeros((len(trials), 3), device=env.device)
                settled_force = torch.zeros((len(trials), 3), device=env.device)
                digit_fraction = torch.zeros((len(trials), 3), device=env.device)
                contact_streak = torch.zeros((len(trials), 3), device=env.device, dtype=torch.int32)
                fixed_fraction = torch.tensor(
                    [(trial.percent or 0) / 100 for trial in trials], device=env.device, dtype=action.dtype
                )
                for index in range(CLOSE_STEPS):
                    if method == "fixed":
                        action[:, 14:] = start_hand + ((index + 1) / CLOSE_STEPS) * fixed_fraction[:, None] * hand_delta
                    else:
                        digit_fraction = torch.where(
                            stable_contact, digit_fraction, torch.clamp(digit_fraction + 0.9 / CLOSE_STEPS, max=0.9)
                        )
                        for digit_index, digit in enumerate(DIGITS):
                            ids = digit_joint_ids[digit]
                            action[:, 14 + torch.tensor(ids, device=env.device)] = (
                                start_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
                            )
                    advance("hand closure")
                    box_slide = torch.maximum(
                        box_slide,
                        torch.linalg.vector_norm(box.data.root_link_pose_w.torch[:, :2] - closure_box_xy, dim=-1),
                    )
                    if method == "contact":
                        forces = _contact_forces(env, torch)
                        peak_force = torch.maximum(peak_force, forces)
                        contact_streak = torch.where(forces >= contact_threshold_n, contact_streak + 1, 0)
                        newly_stable = ~stable_contact & (contact_streak >= 3)
                        force_at_stop = torch.where(newly_stable, forces, force_at_stop)
                        digit_fraction = torch.where(
                            newly_stable, torch.clamp(digit_fraction + 0.02, max=0.9), digit_fraction
                        )
                        stable_contact |= newly_stable
                if method == "contact":
                    for digit_index, digit in enumerate(DIGITS):
                        ids = digit_joint_ids[digit]
                        action[:, 14 + torch.tensor(ids, device=env.device)] = (
                            start_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
                        )
                for _ in range(CLOSE_HOLD_STEPS):
                    advance("grasp settle")
                if method == "contact":
                    forces = _contact_forces(env, torch)
                    settled_force = forces.clone()
                    peak_force = torch.maximum(peak_force, forces)
                    still_touching = forces >= contact_threshold_n
                    thumb_ready = stable_contact[:, 0] & still_touching[:, 0]
                    opposing_ready = (stable_contact[:, 1] & still_touching[:, 1]) | (
                        stable_contact[:, 2] & still_touching[:, 2]
                    )
                    grip_ready = thumb_ready & opposing_ready
                else:
                    grip_ready = torch.ones(len(trials), device=env.device, dtype=torch.bool)

                lift_start = action[:, 7:10].clone()
                lift_delta = torch.tensor([0.0, 0.0, 0.15], device=env.device, dtype=action.dtype)
                for index in range(LIFT_STEPS):
                    action[:, 7:10] = lift_start + ((index + 1) / LIFT_STEPS) * grip_ready[:, None] * lift_delta
                    advance("lift")
                    rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
                    max_rise = torch.maximum(max_rise, rise)
                for _ in range(LIFT_HOLD_STEPS):
                    advance("lift hold", update_lift=True)
                final_rise = box.data.root_link_pose_w.torch[:, 2] - initial_z
                success = (
                    grip_ready & (hold_streak >= math.ceil(0.5 / (cfg.sim.dt * cfg.decimation))) & (final_rise >= 0.10)
                )
                actual_hand = robot.data.joint_pos.torch[:, hand_ids]

            results = []
            for index, trial in enumerate(trials):
                if bool(success[index].item()):
                    failure_reason = ""
                elif not bool(grip_ready[index].item()):
                    failure_reason = "incomplete_contact"
                else:
                    failure_reason = "box_not_held_above_10cm"
                record = {
                    "grasp_id": f"grasp_{trial.grasp_id}",
                    "stage": trial.stage,
                    "repeat": trial.repeat,
                    "closure_percent": trial.percent if method == "fixed" else "",
                    "contact_threshold_n": contact_threshold_n if method == "contact" else "",
                    "success": bool(success[index].item()),
                    "failure_reason": failure_reason,
                    "max_box_rise_m": float(max_rise[index].item()),
                    "final_box_rise_m": float(final_rise[index].item()),
                    "box_slide_during_close_m": float(box_slide[index].item()),
                    "wrist_position_error_m": float(position_error[index].item()),
                    "wrist_rotation_error_deg": float(rotation_error[index].item()),
                    "contact_thumb": bool(stable_contact[index, 0].item()) if method == "contact" else "",
                    "contact_index": bool(stable_contact[index, 1].item()) if method == "contact" else "",
                    "contact_middle": bool(stable_contact[index, 2].item()) if method == "contact" else "",
                    "peak_thumb_force_n": float(peak_force[index, 0].item()) if method == "contact" else "",
                    "peak_index_force_n": float(peak_force[index, 1].item()) if method == "contact" else "",
                    "peak_middle_force_n": float(peak_force[index, 2].item()) if method == "contact" else "",
                    "stop_thumb_force_n": float(force_at_stop[index, 0].item()) if method == "contact" else "",
                    "stop_index_force_n": float(force_at_stop[index, 1].item()) if method == "contact" else "",
                    "stop_middle_force_n": float(force_at_stop[index, 2].item()) if method == "contact" else "",
                    "settled_thumb_force_n": float(settled_force[index, 0].item()) if method == "contact" else "",
                    "settled_index_force_n": float(settled_force[index, 1].item()) if method == "contact" else "",
                    "settled_middle_force_n": float(settled_force[index, 2].item()) if method == "contact" else "",
                    "thumb_target_percent": float(100 * digit_fraction[index, 0].item()) if method == "contact" else "",
                    "index_target_percent": float(100 * digit_fraction[index, 1].item()) if method == "contact" else "",
                    "middle_target_percent": (
                        float(100 * digit_fraction[index, 2].item()) if method == "contact" else ""
                    ),
                }
                for ids in digit_joint_ids.values():
                    for joint_index in ids:
                        name = hand_names[joint_index]
                        record[f"target_{name}_rad"] = float(action[index, 14 + joint_index].item())
                        record[f"actual_{name}_rad"] = float(actual_hand[index, joint_index].item())
                results.append(record)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with result_file.open("w") as file:
                json.dump(results, file)
            print(f"Finished {len(trials)} {method} trials; saved {result_file}", flush=True)
            return results
        finally:
            env.close()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No trial results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_subprocess(
    trials: list[Trial], method: str, launcher_args, video_dir: Path | None = None, contact_threshold_n: float = 0.5
) -> list[dict]:
    """Run one Kit lifetime in a child; Kit exits the process when it closes."""
    prior_videos = {}
    if video_dir is not None:
        for trial in trials:
            prior_videos[trial.label] = set(video_dir.glob(f"{method}_{trial.label}_*.mp4"))
    with tempfile.TemporaryDirectory(prefix="g1_closure_batch_") as work_dir:
        payload_file = Path(work_dir) / "payload.json"
        result_file = Path(work_dir) / "results.json"
        with payload_file.open("w") as file:
            json.dump(
                {
                    "trials": [vars(trial) for trial in trials],
                    "method": method,
                    "launcher_args": vars(launcher_args),
                    "result_file": str(result_file),
                    "video_dir": str(video_dir) if video_dir is not None else None,
                    "contact_threshold_n": contact_threshold_n,
                },
                file,
                default=str,
            )
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(payload_file)], check=True)
        if not result_file.exists():
            raise RuntimeError(f"{method} batch exited without writing results; check the simulation log above")
        with result_file.open() as file:
            results = json.load(file)
        if video_dir is not None:
            for trial, result in zip(trials, results, strict=True):
                videos = set(video_dir.glob(f"{method}_{trial.label}_*.mp4")) - prior_videos[trial.label]
                result["video_file"] = str(max(videos, key=lambda path: path.stat().st_mtime)) if videos else ""
        return results


if __name__ == "__main__":
    from argparse import Namespace

    with Path(sys.argv[1]).open() as file:
        payload = json.load(file)
    run_batch(
        [Trial(**trial) for trial in payload["trials"]],
        payload["method"],
        Namespace(**payload["launcher_args"]),
        Path(payload["result_file"]),
        video_dir=Path(payload["video_dir"]) if payload["video_dir"] else None,
        contact_threshold_n=payload["contact_threshold_n"],
    )
