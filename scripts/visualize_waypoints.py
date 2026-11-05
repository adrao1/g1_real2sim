"""Save a screenshot of FoundationPose waypoints and the estimated camera location."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from evaluate_contact_grasps import configure_item

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, PreviewSurfaceCfg, SphereCfg

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def _matrix(value: object, label: str) -> np.ndarray | None:
    """Read an optional homogeneous transform from a tracking config."""
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
        or any(isinstance(entry, bool) or not isinstance(entry, (int, float)) for row in value for entry in row)
    ):
        raise ValueError(f"{label} must be a 4x4 matrix of numbers")
    matrix = np.asarray(value, dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1)):
        raise ValueError(f"{label} must be a finite homogeneous transform")
    return matrix


def load_waypoints(
    path: Path,
) -> tuple[Path, list[tuple[str, tuple[float, float, float]]], tuple[float, float, float] | None, bool]:
    """Read FoundationPose poses, transforming them to world XYZ when calibrated."""
    path = path.expanduser().resolve(strict=True)
    with path.open() as file:
        tracking = yaml.safe_load(file)
    if not isinstance(tracking, dict):
        raise ValueError(f"Tracking config must be a mapping: {path}")
    if not isinstance(tracking.get("item_config"), str) or not isinstance(tracking.get("pose_dir"), str):
        raise ValueError("Tracking config needs item_config and pose_dir paths")
    world_from_camera = _matrix(tracking.get("world_from_camera"), "world_from_camera")
    foundation_mesh_from_item = _matrix(tracking.get("foundation_mesh_from_item"), "foundation_mesh_from_item")
    if (world_from_camera is None) != (foundation_mesh_from_item is None):
        raise ValueError("world_from_camera and foundation_mesh_from_item must both be set or both be null")
    aligned = world_from_camera is not None
    camera_position = tracking.get("camera_position_world")
    if camera_position is not None:
        if (
            not isinstance(camera_position, list)
            or len(camera_position) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in camera_position)
            or not np.isfinite(camera_position).all()
        ):
            raise ValueError("camera_position_world must be three finite XYZ numbers")
        camera_position = tuple(float(value) for value in camera_position)
    if aligned:
        matrix_position = tuple(float(value) for value in world_from_camera[:3, 3])
        if camera_position is not None and not np.allclose(camera_position, matrix_position, atol=1e-3):
            raise ValueError("camera_position_world must match the translation in world_from_camera")
        camera_position = matrix_position
    item_path = (path.parent / tracking["item_config"]).resolve(strict=True)
    pose_dir = (path.parent / tracking["pose_dir"]).resolve(strict=True)
    if not pose_dir.is_dir():
        raise ValueError(f"Pose path must be a directory: {pose_dir}")
    files = list(pose_dir.glob("*.txt"))
    if not files or any(not file.stem.isdigit() for file in files):
        raise ValueError(f"Expected numbered FoundationPose .txt files in {pose_dir}")
    waypoints = []
    for file in sorted(files, key=lambda file: int(file.stem)):
        pose = np.loadtxt(file)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], (0, 0, 0, 1)):
            raise ValueError(f"Invalid 4x4 pose: {file}")
        if aligned:
            pose = world_from_camera @ pose @ foundation_mesh_from_item
        waypoints.append((file.stem, tuple(float(value) for value in pose[:3, 3])))
    return item_path, waypoints, camera_position, aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument(
        "--tracking-config", required=True, type=Path, help="Tracking YAML with item and pose directory"
    )
    parser.add_argument("--output", type=Path, help="PNG path; defaults to outputs/waypoint_screenshots")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    cfg = PickPlaceEnvCfg()
    try:
        item_path, waypoints, camera_position, aligned = load_waypoints(args.tracking_config)
        item_args = argparse.Namespace(item_config=item_path, item_position=None, item_rotation=None)
        item = configure_item(cfg, item_args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or Path("outputs/waypoint_screenshots") / (
        f"{item.name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}.png"
    )
    if output.suffix.lower() != ".png" or output.exists():
        parser.error("--output must be a new .png file")
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        width=960,
        height=720,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
    )

    with launch_simulation(cfg, args):
        import torch

        from isaaclab.scene import InteractiveScene
        from isaaclab.sensors.camera.utils import save_images_to_file
        from isaaclab.sim import SimulationContext, create_prim, spawn_sphere

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        create_prim("/World/Waypoints", prim_type="Xform")
        marker = SphereCfg(radius=0.02, visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)))
        for frame_id, position in waypoints:
            spawn_sphere(f"/World/Waypoints/Frame_{frame_id}", marker, translation=position)
        if camera_position is not None:
            camera_marker = SphereCfg(
                radius=0.04,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
            )
            spawn_sphere("/World/EstimatedCameraPosition", camera_marker, translation=camera_position)
        sim.reset()
        scene.reset()
        camera = scene["camera"]
        origin = scene.env_origins
        focus = torch.tensor(item.position, device=cfg.sim.device)
        if camera_position is not None:
            focus = (focus + torch.tensor(camera_position, device=cfg.sim.device)) / 2
        view_offset = (1.2, 1.2, 0.8) if camera_position is not None else (0.9, 0.9, 0.65)
        camera.set_world_poses_from_view(
            origin + focus + torch.tensor(view_offset, device=cfg.sim.device),
            origin + focus,
        )
        robot = scene["robot"]
        robot.set_joint_position_target(robot.data.default_joint_pos.torch)
        for _ in range(100):
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())

        output.parent.mkdir(parents=True, exist_ok=True)
        save_images_to_file(camera.data.output["rgb"].torch.float() / 255.0, str(output))
        frame = "world-frame" if aligned else "raw camera-frame"
        print(f"Saved {len(waypoints)} {frame} waypoints to {output.resolve()}", flush=True)
        if camera_position is not None:
            print(f"Blue camera-position marker: {camera_position}", flush=True)


if __name__ == "__main__":
    main()
