"""Evaluate object-relative grasps with contact-guided G1 finger closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.visualizers import ImageViewCfg

from g1_real2sim.assets import G1_REAL2SIM_ASSETS_DIR
from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg

DEFAULT_GRASPS = G1_REAL2SIM_ASSETS_DIR / "grasps/cracker_box_grasps.yml"
DEFAULT_OUTPUT_ROOT = Path("outputs/grasp_evaluation/contact_grasps")
GRIPPER_NAME = "unitree_g1"
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
class Item:
    name: str
    usd_path: str
    scale: float
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    grasp_file: Path | None


def _numbers(value, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    try:
        numbers = tuple(float(number) for number in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {count} numbers") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{label} must contain finite numbers")
    return numbers


def _rotation(value, label: str) -> tuple[float, float, float, float]:
    rotation = _numbers(value, 4, label)
    norm = math.sqrt(sum(component * component for component in rotation))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"{label} must be a unit quaternion in xyzw order")
    return tuple(component / norm for component in rotation)


def configure_item(cfg: PickPlaceEnvCfg, args: argparse.Namespace) -> Item:
    """Resolve one item and apply it to the scene before simulation starts."""
    scene_item = cfg.scene.object
    name = "cracker_box"
    usd_path = scene_item.spawn.usd_path
    scale = scene_item.spawn.scale[0] if scene_item.spawn.scale is not None else 1.0
    position = scene_item.init_state.pos
    rotation = scene_item.init_state.rot
    grasp_file = DEFAULT_GRASPS
    if args.item_config is not None:
        preset = args.item_config.expanduser().resolve(strict=True)
        with preset.open() as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Item preset must be a mapping: {preset}")
        name = data.get("name", preset.stem)
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("Item name must use only letters, digits, underscores, or hyphens")
        if not isinstance(data.get("usd_path"), str):
            raise ValueError(f"Item preset needs a local usd_path: {preset}")
        path = Path(data["usd_path"]).expanduser()
        usd_path = (
            str((preset.parent / path).resolve(strict=True))
            if not path.is_absolute()
            else str(path.resolve(strict=True))
        )
        if Path(usd_path).suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
            raise ValueError(f"Item must be a USD asset: {usd_path}")
        scale = float(data.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Item scale must be positive and finite")
        position = _numbers(data.get("position"), 3, "item position")
        rotation = _rotation(data.get("rotation_xyzw"), "item rotation_xyzw")
        grasp_file = None
        if data.get("grasp_file") is not None:
            path = Path(data["grasp_file"]).expanduser()
            grasp_file = (preset.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if args.item_position is not None:
        position = _numbers(args.item_position, 3, "--item-position")
    if args.item_rotation is not None:
        rotation = _rotation(args.item_rotation, "--item-rotation")
    cfg.scene.set_item(usd_path, position, rotation, scale)
    resolved = cfg.scene.object
    return Item(name, resolved.spawn.usd_path, scale, resolved.init_state.pos, resolved.init_state.rot, grasp_file)


def load_grasps(path: Path) -> dict[int, dict]:
    """Validate the Isaac grasp YAML and keep its numeric grasp IDs."""
    with path.open() as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or data.get("format") != "isaac_grasp" or data.get("format_version") != 1.0:
        raise ValueError(f"Expected Isaac grasp format version 1.0 in {path}")
    if not isinstance(data.get("grasps"), dict) or not data["grasps"]:
        raise ValueError(f"Expected a nonempty grasps mapping in {path}")
    grasps = {}
    for key, grasp in data["grasps"].items():
        match = re.fullmatch(r"grasp_(\d+)", str(key))
        if match is None or not isinstance(grasp, dict):
            raise ValueError(f"Invalid grasp entry {key!r} in {path}")
        index = int(match.group(1))
        if index in grasps:
            raise ValueError(f"Duplicate numeric grasp ID {index} in {path}")
        position = _numbers(grasp.get("position"), 3, f"{key} position")
        orientation = grasp.get("orientation")
        if not isinstance(orientation, dict):
            raise ValueError(f"{key} needs an orientation mapping")
        quaternion = _rotation(
            (*_numbers(orientation.get("xyz"), 3, f"{key} orientation.xyz"), orientation.get("w")), f"{key} orientation"
        )
        grasps[index] = {"position": position, "quaternion_xyzw": quaternion}
    return dict(sorted(grasps.items()))


def select_grasps(args: argparse.Namespace, grasps: dict[int, dict], item: Item) -> list[int]:
    if args.grasps is not None:
        selected = args.grasps
    elif args.successful_from is not None:
        with args.successful_from.open(newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"grasp_id", "success"}.issubset(reader.fieldnames):
                raise ValueError("Success CSV needs grasp_id and success columns")
            rows = list(reader)
        if not rows:
            raise ValueError("Success CSV is empty")
        if "item_name" in rows[0]:
            if any(row["item_name"] != item.name for row in rows):
                raise ValueError("Success CSV belongs to another item")
        elif item.name != "cracker_box":
            raise ValueError("A legacy success CSV without item_name can only select cracker-box grasps")
        selected = list(
            dict.fromkeys(
                int(row["grasp_id"].removeprefix("grasp_")) for row in rows if row["success"].strip().lower() == "true"
            )
        )
    else:
        selected = list(grasps)
    if not selected or len(set(selected)) != len(selected) or any(index not in grasps for index in selected):
        raise ValueError("Select distinct grasp IDs present in the selected grasp YAML")
    return selected


def _grasp_cache_key(usd_path: Path, scale: float, generator_script: Path) -> str:
    digest = hashlib.sha256()
    for path in (usd_path, generator_script):
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    digest.update(f"{scale:.12g}:{GRIPPER_NAME}:top100".encode())
    return digest.hexdigest()[:16]


def obtain_grasps(item: Item, args: argparse.Namespace) -> Path:
    """Use a supplied file or generate and cache matching grasps before Kit starts."""
    if args.grasp_file is not None:
        if args.regenerate_grasps:
            raise ValueError("--grasp-file and --regenerate-grasps cannot be combined")
        path = args.grasp_file.expanduser().resolve(strict=True)
        load_grasps(path)
        return path
    if item.grasp_file is not None and not args.regenerate_grasps:
        path = item.grasp_file.resolve(strict=True)
        load_grasps(path)
        return path
    usd_path = Path(item.usd_path)
    if not usd_path.is_file():
        raise ValueError("GraspGenX needs a local item USD; provide --grasp-file for a remote USD")
    root = (args.graspgenx_root or Path(__file__).resolve().parents[2] / "GraspGenX").expanduser().resolve()
    python = root / ".venv/bin/python"
    script = root / "scripts/demo_object_mesh.py"
    if not python.is_file() or not script.is_file():
        raise ValueError(f"GraspGenX checkout with .venv not found at {root}; use --graspgenx-root")
    cache_dir = (Path("outputs/grasp_evaluation/grasp_cache") / item.name).resolve()
    cache_path = cache_dir / f"{_grasp_cache_key(usd_path, item.scale, script)}.yml"
    if cache_path.is_file() and not args.regenerate_grasps:
        load_grasps(cache_path)
        return cache_path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="graspgenx_", dir=cache_dir) as temp_dir:
        output = Path(temp_dir) / "grasps.yml"
        command = [
            str(python),
            str(script),
            "--mesh_file",
            str(usd_path),
            "--mesh_scale",
            str(item.scale),
            "--gripper_name",
            GRIPPER_NAME,
            "--grasp_threshold",
            "-1.0",
            "--return_topk",
            "--topk_num_grasps",
            "100",
            "--no-visualization",
            "--output_file",
            str(output),
        ]
        print(f"Generating grasps for {item.name} with GraspGenX...", flush=True)
        subprocess.run(command, cwd=root, check=True)
        if not output.is_file():
            raise RuntimeError("GraspGenX produced no grasp YAML")
        load_grasps(output)
        os.replace(output, cache_path)
    return cache_path.resolve()


def configure_contact_sensors(cfg: PickPlaceEnvCfg) -> None:
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.object.spawn.activate_contact_sensors = True
    for links in CONTACT_LINKS.values():
        for link in links:
            setattr(
                cfg.scene,
                f"contact_{link}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
                    filter_prim_paths_expr=[cfg.scene.object.prim_path],
                    update_period=0.0,
                ),
            )


def contact_forces(env, torch):
    """Return box-filtered normal-force magnitudes for thumb, index, and middle."""
    digit_forces = []
    for digit in DIGITS:
        link_forces = []
        for link in CONTACT_LINKS[digit]:
            matrix = env.scene[f"contact_{link}"].data.normal_force_matrix_w
            if matrix is None or matrix.torch.ndim != 4 or matrix.torch.shape[1:] != (1, 1, 3):
                raise RuntimeError(f"Filtered item contact data is unavailable for {link}")
            link_forces.append(torch.linalg.vector_norm(matrix.torch[:, 0, 0, :], dim=-1))
        digit_forces.append(torch.stack(link_forces, dim=1).amax(dim=1))
    return torch.stack(digit_forces, dim=1)


def configure_video(cfg: PickPlaceEnvCfg, video_dir: Path) -> None:
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        width=320,
        height=240,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
    )
    cfg.video_recorders = [
        VideoRecorderCfg(
            source=ImageViewCfg(source="camera", envs=(index,)),
            output_dir=str(video_dir),
            output_filename_prefix=f"slot_{index:02d}",
            video_length=TOTAL_STEPS,
            video_interval=TOTAL_STEPS,
            frame_stride=2,
            fps=25,
        )
        for index in range(cfg.scene.num_envs)
    ]


def run_attempt(
    env,
    cfg: PickPlaceEnvCfg,
    grasps: dict[int, dict],
    grasp_ids: list[int],
    active: int,
    repeat: int,
    item_name: str,
    threshold_n: float,
) -> list[dict]:
    """Run exactly TOTAL_STEPS for one fresh reset of every environment."""
    import torch

    from isaaclab.utils.math import (
        combine_frame_transforms,
        quat_error_magnitude,
        quat_from_euler_xyz,
        subtract_frame_transforms,
    )

    robot = env.scene["robot"]
    item = env.scene["object"]
    origins = env.scene.env_origins
    wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    hand_ids, hand_names = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)
    wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
    wrist_poses[..., :3] -= origins[:, None, :]
    action = torch.zeros_like(env.action_manager.action)
    action[:, :14] = wrist_poses.flatten(start_dim=1)
    action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]
    max_rise = torch.zeros(len(grasp_ids), device=env.device)
    hold_streak = torch.zeros(len(grasp_ids), device=env.device, dtype=torch.int32)

    def advance(stage: str, update_lift: bool = False) -> None:
        nonlocal max_rise, hold_streak
        _, _, terminated, truncated, _ = env.step(action)
        if bool(torch.any(terminated | truncated)):
            raise RuntimeError(f"An environment reset during {stage}")
        if update_lift:
            rise = item.data.root_link_pose_w.torch[:, 2] - initial_z
            max_rise = torch.maximum(max_rise, rise)
            hold_streak = torch.where(rise >= 0.10, hold_streak + 1, 0)

    with torch.inference_mode():
        for _ in range(SETTLE_STEPS):
            advance("item settle")
        object_pose = item.data.root_link_pose_w.torch.clone()
        initial_z = object_pose[:, 2].clone()
        grasp_pos = torch.tensor(
            [grasps[index]["position"] for index in grasp_ids], device=env.device, dtype=object_pose.dtype
        )
        grasp_quat = torch.tensor(
            [grasps[index]["quaternion_xyzw"] for index in grasp_ids], device=env.device, dtype=object_pose.dtype
        )
        grasp_pos_w, grasp_quat_w = combine_frame_transforms(
            object_pose[:, :3], object_pose[:, 3:7], grasp_pos, grasp_quat
        )
        root_to_palm_pos = torch.tensor([[0.07, 0.0, 0.02]], device=env.device, dtype=object_pose.dtype).expand(
            len(grasp_ids), -1
        )
        root_to_palm_quat = quat_from_euler_xyz(
            torch.tensor([1.5708], device=env.device),
            torch.tensor([0.0], device=env.device),
            torch.tensor([3.14159], device=env.device),
        ).expand(len(grasp_ids), -1)
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
        for step in range(APPROACH_STEPS):
            fraction = (step + 1) / APPROACH_STEPS
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
        closure_item_xy = item.data.root_link_pose_w.torch[:, :2].clone()
        item_slide = torch.zeros(len(grasp_ids), device=env.device)
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
        digit_action_ids = {digit: 14 + torch.tensor(ids, device=env.device) for digit, ids in digit_joint_ids.items()}
        stable_contact = torch.zeros((len(grasp_ids), 3), device=env.device, dtype=torch.bool)
        peak_force = torch.zeros((len(grasp_ids), 3), device=env.device)
        force_at_stop = torch.zeros((len(grasp_ids), 3), device=env.device)
        digit_fraction = torch.zeros((len(grasp_ids), 3), device=env.device)
        contact_streak = torch.zeros((len(grasp_ids), 3), device=env.device, dtype=torch.int32)
        for _ in range(CLOSE_STEPS):
            digit_fraction = torch.where(
                stable_contact, digit_fraction, torch.clamp(digit_fraction + 0.9 / CLOSE_STEPS, max=0.9)
            )
            for digit_index, digit in enumerate(DIGITS):
                ids = digit_joint_ids[digit]
                action[:, digit_action_ids[digit]] = (
                    start_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
                )
            advance("hand closure")
            item_slide = torch.maximum(
                item_slide, torch.linalg.vector_norm(item.data.root_link_pose_w.torch[:, :2] - closure_item_xy, dim=-1)
            )
            forces = contact_forces(env, torch)
            peak_force = torch.maximum(peak_force, forces)
            contact_streak = torch.where(forces >= threshold_n, contact_streak + 1, 0)
            newly_stable = ~stable_contact & (contact_streak >= 3)
            force_at_stop = torch.where(newly_stable, forces, force_at_stop)
            digit_fraction = torch.where(newly_stable, torch.clamp(digit_fraction + 0.02, max=0.9), digit_fraction)
            stable_contact |= newly_stable
        for digit_index, digit in enumerate(DIGITS):
            ids = digit_joint_ids[digit]
            action[:, digit_action_ids[digit]] = (
                start_hand[:, ids] + digit_fraction[:, digit_index, None] * hand_delta[:, ids]
            )
        for _ in range(CLOSE_HOLD_STEPS):
            advance("grasp settle")
        settled_force = contact_forces(env, torch)
        peak_force = torch.maximum(peak_force, settled_force)
        still_touching = settled_force >= threshold_n
        thumb_ready = stable_contact[:, 0] & still_touching[:, 0]
        opposing_ready = (stable_contact[:, 1] & still_touching[:, 1]) | (stable_contact[:, 2] & still_touching[:, 2])
        grip_ready = thumb_ready & opposing_ready
        lift_start = action[:, 7:10].clone()
        lift_delta = torch.tensor([0.0, 0.0, 0.15], device=env.device, dtype=action.dtype)
        for step in range(LIFT_STEPS):
            action[:, 7:10] = lift_start + ((step + 1) / LIFT_STEPS) * grip_ready[:, None] * lift_delta
            advance("lift")
            rise = item.data.root_link_pose_w.torch[:, 2] - initial_z
            max_rise = torch.maximum(max_rise, rise)
        for _ in range(LIFT_HOLD_STEPS):
            advance("lift hold", update_lift=True)
        final_rise = item.data.root_link_pose_w.torch[:, 2] - initial_z
        success = grip_ready & (hold_streak >= math.ceil(0.5 / (cfg.sim.dt * cfg.decimation))) & (final_rise >= 0.10)
        actual_hand = robot.data.joint_pos.torch[:, hand_ids]

    rows = []
    for index in range(active):
        if bool(success[index].item()):
            reason = ""
        elif not bool(grip_ready[index].item()):
            reason = "incomplete_contact"
        else:
            reason = "item_not_held_above_10cm"
        row = {
            "item_name": item_name,
            "grasp_id": f"grasp_{grasp_ids[index]}",
            "repeat": repeat,
            "success": bool(success[index].item()),
            "failure_reason": reason,
            "contact_threshold_n": threshold_n,
            "max_item_rise_m": float(max_rise[index].item()),
            "final_item_rise_m": float(final_rise[index].item()),
            "item_slide_during_close_m": float(item_slide[index].item()),
            "wrist_position_error_m": float(position_error[index].item()),
            "wrist_rotation_error_deg": float(rotation_error[index].item()),
            "video_file": "",
        }
        for digit_index, digit in enumerate(DIGITS):
            row[f"contact_{digit}"] = bool(stable_contact[index, digit_index].item())
            row[f"peak_{digit}_force_n"] = float(peak_force[index, digit_index].item())
            row[f"stop_{digit}_force_n"] = float(force_at_stop[index, digit_index].item())
            row[f"settled_{digit}_force_n"] = float(settled_force[index, digit_index].item())
            row[f"{digit}_target_percent"] = float(100 * digit_fraction[index, digit_index].item())
        for ids in digit_joint_ids.values():
            for joint_index in ids:
                name = hand_names[joint_index]
                row[f"target_{name}_rad"] = float(action[index, 14 + joint_index].item())
                row[f"actual_{name}_rad"] = float(actual_hand[index, joint_index].item())
        rows.append(row)
    return rows


def finalize_videos(video_dir: Path, rows: list[dict], num_envs: int, clip_index: int) -> list[str]:
    """Keep only this batch's successful clips and give them stable trial names."""
    missing = []
    for slot in range(num_envs):
        clip = video_dir / f"slot_{slot:02d}_{clip_index:04d}.mp4"
        if slot < len(rows) and rows[slot]["success"]:
            if not clip.is_file():
                missing.append(str(clip))
                continue
            target = video_dir / f"{rows[slot]['grasp_id']}_repeat_{rows[slot]['repeat']}.mp4"
            clip.rename(target)
            rows[slot]["video_file"] = str(target)
        else:
            clip.unlink(missing_ok=True)
    return missing


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _run_directory(item: Item, requested: Path | None) -> Path:
    if requested is None:
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        path = DEFAULT_OUTPUT_ROOT / item.name / name
    else:
        path = requested.expanduser()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def run_evaluation(
    cfg: PickPlaceEnvCfg,
    args: argparse.Namespace,
    item: Item,
    grasps: dict[int, dict],
    selected: list[int],
    grasp_file: Path,
    output_dir: Path,
) -> None:
    video_dir = output_dir / "videos" if not args.no_video else None
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = min(args.batch_size, len(selected))
    cfg.episode_length_s = 30.0
    configure_contact_sensors(cfg)
    if video_dir is not None:
        configure_video(cfg, video_dir)
    manifest = {
        "item_name": item.name,
        "object_usd": item.usd_path,
        "object_scale": item.scale,
        "object_start_position": item.position,
        "object_start_rotation_xyzw": item.rotation_xyzw,
        "grasp_file": str(grasp_file),
        "grasp_sha256": hashlib.sha256(grasp_file.read_bytes()).hexdigest(),
        "selected_grasps": [f"grasp_{index}" for index in selected],
        "repeats": args.repeats,
        "batch_size": cfg.scene.num_envs,
        "contact_threshold_n": args.contact_threshold_n,
        "video_enabled": video_dir is not None,
        "status": "started",
    }
    manifest_file = output_dir / "run.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with launch_simulation(cfg, args), tempfile.TemporaryDirectory(prefix="g1_urdf_") as urdf_dir:
        import torch

        from isaaclab.controllers.utils import convert_usd_to_urdf
        from isaaclab.envs import ManagerBasedRLEnv

        urdf_path, _ = convert_usd_to_urdf(cfg.scene.robot.spawn.usd_path, urdf_dir)
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
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            num_envs = cfg.scene.num_envs
            clip_index = 0
            for repeat in range(1, args.repeats + 1):
                for offset in range(0, len(selected), num_envs):
                    batch = selected[offset : offset + num_envs]
                    padded = batch + [batch[0]] * (num_envs - len(batch))
                    env.reset()
                    if video_dir is not None:
                        origins = env.scene.env_origins
                        focus = torch.tensor(item.position, device=env.device)
                        env.scene["camera"].set_world_poses_from_view(
                            origins + focus + torch.tensor((0.9, 0.9, 0.65), device=env.device),
                            origins + focus,
                        )
                    print(f"Repeat {repeat}: grasps {batch[0]}–{batch[-1]} ({len(batch)} active)", flush=True)
                    rows = run_attempt(
                        env, cfg, grasps, padded, len(batch), repeat, item.name, args.contact_threshold_n
                    )
                    missing_videos = (
                        finalize_videos(video_dir, rows, num_envs, clip_index) if video_dir is not None else []
                    )
                    results.extend(rows)
                    write_csv(output_dir / "trials.csv", results)
                    if missing_videos:
                        raise RuntimeError(f"Successful trials are missing video clips: {', '.join(missing_videos)}")
                    clip_index += 1
            manifest["status"] = "complete"
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            successes = sum(row["success"] for row in results)
            print(f"Contact-guided lifts: {successes}/{len(results)}", flush=True)
            print(f"Results saved to {output_dir / 'trials.csv'}", flush=True)
            if video_dir is not None:
                print(f"Successful videos saved under {video_dir}", flush=True)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument(
        "--item-config", type=Path, help="YAML item preset: USD, scale, position, rotation, optional grasps"
    )
    parser.add_argument("--item-position", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--item-rotation", nargs=4, type=float, metavar=("X", "Y", "Z", "W"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--grasps", nargs="+", type=int, metavar="ID")
    selection.add_argument("--successful-from", type=Path, help="Select successful grasp IDs from a CSV")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--grasp-file", type=Path, help="Use this Isaac grasp YAML instead of the item preset or cache")
    parser.add_argument("--regenerate-grasps", action="store_true", help="Refresh the GraspGenX cache")
    parser.add_argument("--graspgenx-root", type=Path, help="GraspGenX checkout with its own .venv")
    parser.add_argument("--batch-size", type=int, help="Parallel environments: default 25 with video, 100 without")
    parser.add_argument("--no-video", action="store_true", help="Disable default recording of successful attempts")
    parser.add_argument("--contact-threshold-n", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, help="New or empty directory for this run")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.contact_threshold_n) or args.contact_threshold_n <= 0:
        parser.error("--repeats and --contact-threshold-n must be positive")
    if args.batch_size is None:
        args.batch_size = 100 if args.no_video else 25
    max_batch_size = 100 if args.no_video else 25
    if not 1 <= args.batch_size <= max_batch_size:
        parser.error(f"--batch-size must be between 1 and {max_batch_size}")
    if not args.no_video:
        try:
            import moviepy  # noqa: F401
        except ImportError:
            parser.error("Default video requires: uv run --extra isaacsim --extra video ... (or use --no-video)")
    try:
        cfg = PickPlaceEnvCfg()
        item = configure_item(cfg, args)
        grasp_file = obtain_grasps(item, args)
        grasps = load_grasps(grasp_file)
        selected = select_grasps(args, grasps, item)
        output_dir = _run_directory(item, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Item: {item.name}; USD: {item.usd_path}; scale: {item.scale}; grasps: {grasp_file}", flush=True)
    run_evaluation(cfg, args, item, grasps, selected, grasp_file, output_dir)


if __name__ == "__main__":
    main()
