"""Save one tabletop screenshot per tracked item pose, without robot motion."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
from evaluate_contact_grasps import configure_item

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import quat_from_matrix

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg
from g1_real2sim.tracking import load_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--tracking-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="New or empty image directory")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    try:
        item_path, frame_numbers, world_poses = load_tracking(args.tracking_config)
        positions = world_poses[:, :3, 3]
        quaternions = quat_from_matrix(torch.as_tensor(world_poses[:, :3, :3], dtype=torch.float64))
        if not bool(torch.isfinite(quaternions).all()):
            raise ValueError("A tracked rotation could not be converted to a quaternion")
        first_position = tuple(float(value) for value in positions[0])
        first_quaternion = tuple(float(value) for value in quaternions[0])
        cfg = PickPlaceEnvCfg()
        item = configure_item(
            cfg,
            argparse.Namespace(item_config=item_path, item_position=first_position, item_rotation=first_quaternion),
        )
        output_dir = args.output_dir or Path("outputs/tracking_pose_screenshots") / args.tracking_config.stem / (
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        )
        output_dir = output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {output_dir}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

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
    focus_position = (positions.min(axis=0) + positions.max(axis=0)) / 2
    print(f"Rendering {len(frame_numbers)} poses of {item.name} to {output_dir}", flush=True)

    with launch_simulation(cfg, args):
        from isaaclab.scene import InteractiveScene
        from isaaclab.sensors.camera.utils import save_images_to_file
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        sim.reset()
        scene.reset()
        camera = scene["camera"]
        origin = scene.env_origins
        focus = torch.as_tensor(focus_position, device=cfg.sim.device, dtype=torch.float32)
        camera.set_world_poses_from_view(
            origin + focus + torch.tensor((0.9, 0.9, 0.65), device=cfg.sim.device),
            origin + focus,
        )
        item_object = scene["object"]
        device_poses = torch.cat(
            (
                torch.as_tensor(positions, device=cfg.sim.device, dtype=torch.float32),
                quaternions.to(device=cfg.sim.device, dtype=torch.float32),
            ),
            dim=1,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, frame_number in enumerate(frame_numbers):
            item_object.write_root_pose_to_sim_index(root_pose=device_poses[index : index + 1])
            sim.forward()
            sim.render()
            scene.update(sim.get_physics_dt())
            image = camera.data.output["rgb"].torch
            image_path = output_dir / f"frame_{int(frame_number):06d}.png"
            save_images_to_file(image.float() / 255.0, str(image_path))
            if (index + 1) % 25 == 0 or index + 1 == len(frame_numbers):
                print(f"Saved {index + 1}/{len(frame_numbers)}: {image_path}", flush=True)


if __name__ == "__main__":
    main()
