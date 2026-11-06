"""Save a screenshot of FoundationPose positions and camera pose."""

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
) -> tuple[Path, list[tuple[str, np.ndarray]], np.ndarray | None]:
    """Read FoundationPose poses, transforming full poses to world when aligned."""
    path = path.expanduser().resolve(strict=True)
    with path.open() as file:
        tracking = yaml.safe_load(file)
    if not isinstance(tracking, dict):
        raise ValueError(f"Tracking config must be a mapping: {path}")
    if not isinstance(tracking.get("item_config"), str) or not isinstance(tracking.get("pose_dir"), str):
        raise ValueError("Tracking config needs item_config and pose_dir paths")
    world_from_camera = _matrix(tracking.get("world_from_camera"), "world_from_camera")
    aligned = world_from_camera is not None
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
            pose = world_from_camera @ pose
        waypoints.append((file.stem, pose))
    return item_path, waypoints, world_from_camera


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
        import torch

        from isaaclab.utils.math import quat_from_matrix

        item_path, waypoints, world_from_camera = load_waypoints(args.tracking_config)
        camera_position = (
            tuple(float(value) for value in world_from_camera[:3, 3]) if world_from_camera is not None else None
        )
        first_pose = waypoints[0][1]
        quaternion = quat_from_matrix(torch.as_tensor(first_pose[:3, :3], dtype=torch.float64))
        if not bool(torch.isfinite(quaternion).all()):
            raise ValueError(f"First tracked rotation cannot be converted to a quaternion: {args.tracking_config}")
        item_args = argparse.Namespace(
            item_config=item_path,
            item_position=tuple(float(value) for value in first_pose[:3, 3]),
            item_rotation=tuple(float(value) for value in quaternion),
        )
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
        from pxr import Gf, UsdGeom

        from isaaclab.scene import InteractiveScene
        from isaaclab.sensors.camera.utils import save_images_to_file
        from isaaclab.sim import SimulationContext, create_prim, get_current_stage, spawn_sphere

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        create_prim("/World/Waypoints", prim_type="Xform")
        positions = [tuple(float(value) for value in pose[:3, 3]) for _, pose in waypoints]
        stage = get_current_stage()
        waypoint_color = (0.05, 0.45, 0.85)

        def draw_axes(path: str, pose: np.ndarray, length: float, width: float) -> None:
            """Draw RGB XYZ arrows from the translation and rotation of a pose."""
            center = pose[:3, 3]
            for axis, (label, color) in enumerate(
                (("X", (1.0, 0.1, 0.1)), ("Y", (0.1, 0.85, 0.1)), ("Z", (0.1, 0.35, 1.0)))
            ):
                direction = pose[:3, axis]
                direction = direction / np.linalg.norm(direction)
                side = pose[:3, (axis + 1) % 3]
                side = side / np.linalg.norm(side)
                tip = center + length * direction
                wing = tip - 0.25 * length * direction
                points = (
                    center,
                    tip,
                    wing + 0.12 * length * side,
                    tip,
                    wing - 0.12 * length * side,
                )
                curve = UsdGeom.BasisCurves.Define(stage, f"{path}_{label}")
                curve.CreateTypeAttr(UsdGeom.Tokens.linear)
                curve.CreateCurveVertexCountsAttr([2, 3])
                curve.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
                curve.CreateWidthsAttr([width])
                curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1))]
        sample_indices = []
        if distance[-1] > 0:
            for target in np.linspace(0, distance[-1], 12)[1:-1]:
                index = int(np.searchsorted(distance, target))
                if all(
                    np.linalg.norm(np.subtract(positions[index], positions[other])) >= 0.022
                    for other in (0, *sample_indices, len(positions) - 1)
                ):
                    sample_indices.append(index)
        for index in sample_indices:
            marker = SphereCfg(
                radius=0.009,
                visual_material=PreviewSurfaceCfg(diffuse_color=waypoint_color),
            )
            spawn_sphere(f"/World/Waypoints/Frame_{waypoints[index][0]}", marker, translation=positions[index])

        start_marker = SphereCfg(
            radius=0.009,
            visual_material=PreviewSurfaceCfg(diffuse_color=waypoint_color),
        )
        spawn_sphere("/World/Waypoints/Start", start_marker, translation=positions[0])
        if len(positions) > 1:
            end_marker = SphereCfg(
                radius=0.009,
                visual_material=PreviewSurfaceCfg(diffuse_color=waypoint_color),
            )
            spawn_sphere("/World/Waypoints/End", end_marker, translation=positions[-1])
        if camera_position is not None:
            camera_marker = SphereCfg(
                radius=0.025,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
            )
            spawn_sphere("/World/EstimatedCameraPosition", camera_marker, translation=camera_position)
            draw_axes("/World/EstimatedCameraOrientation", world_from_camera, length=0.12, width=0.005)
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
        frame = "world-frame" if world_from_camera is not None else "raw camera-frame"
        print(f"Saved {len(waypoints)} {frame} waypoints to {output.resolve()}", flush=True)
        print("Object waypoints: light blue spheres", flush=True)
        if camera_position is not None:
            print(f"Blue camera-position marker: {camera_position}", flush=True)
            print("Camera orientation arrows: red +X, green +Y, blue +Z", flush=True)


if __name__ == "__main__":
    main()
