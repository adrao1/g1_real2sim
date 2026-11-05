"""Save a screenshot of a waypoint marker at an item's start position."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evaluate_contact_grasps import configure_item

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, PreviewSurfaceCfg, SphereCfg

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--item-config", required=True, type=Path, help="Item preset YAML")
    parser.add_argument("--output", type=Path, help="PNG path; defaults to outputs/waypoint_screenshots")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    cfg = PickPlaceEnvCfg()
    args.item_position = None
    args.item_rotation = None
    try:
        item = configure_item(cfg, args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or Path("outputs/waypoint_screenshots") / (
        f"{item.name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}.png"
    )
    if output.suffix.lower() != ".png" or output.exists():
        parser.error("--output must be a new .png file")
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.scene.waypoint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Waypoint",
        init_state=AssetBaseCfg.InitialStateCfg(pos=item.position),
        spawn=SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    )
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
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
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
        print(f"Saved waypoint screenshot to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
