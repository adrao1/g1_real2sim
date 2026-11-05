"""Save a screenshot of raw FoundationPose positions in the tabletop scene."""

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


def load_waypoints(path: Path) -> tuple[Path, list[tuple[str, tuple[float, float, float]]]]:
    """Read each FoundationPose translation as a simulator world position."""
    path = path.expanduser().resolve(strict=True)
    with path.open() as file:
        tracking = yaml.safe_load(file)
    if not isinstance(tracking, dict):
        raise ValueError(f"Tracking config must be a mapping: {path}")
    if not isinstance(tracking.get("item_config"), str) or not isinstance(tracking.get("pose_dir"), str):
        raise ValueError("Tracking config needs item_config and pose_dir paths")
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
        waypoints.append((file.stem, tuple(float(value) for value in pose[:3, 3])))
    return item_path, waypoints


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
        item_path, waypoints = load_waypoints(args.tracking_config)
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
        sim.reset()
        scene.reset()
        camera = scene["camera"]
        origin = scene.env_origins
        focus = torch.tensor(item.position, device=cfg.sim.device)
        camera.set_world_poses_from_view(
            origin + focus + torch.tensor((0.9, 0.9, 0.65), device=cfg.sim.device),
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
        print(f"Saved {len(waypoints)} raw camera-frame waypoints to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
