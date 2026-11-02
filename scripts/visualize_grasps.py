"""Move 100 G1 right wrists 0.20 m along +X and save one image per environment."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    cfg = PickPlaceEnvCfg()
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = 100
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

        print(f"Converting G1 USD to URDF once for all {cfg.scene.num_envs} IK controllers...", flush=True)
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

            start_x = action[:, 7].clone()
            target_x = start_x + 0.20
            print(f"Right wrist start X: {start_x[0].item():.4f} m")
            print(f"Right wrist target X: {target_x[0].item():.4f} m")

            with torch.inference_mode():
                for step in range(100):
                    action[:, 7] = start_x + (step + 1) / 100 * 0.20
                    env.step(action)
                    if (step + 1) % 25 == 0:
                        print(f"Move step {step + 1}/100", flush=True)
                print("Holding final pose for 100 steps...", flush=True)
                for _ in range(100):
                    env.step(action)

            actual_x = robot.data.body_link_pose_w.torch[:, wrist_ids[1], 0] - origins[:, 0]
            print("Capturing camera images...", flush=True)
            images = camera.data.output["rgb"].torch
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for env_id in range(cfg.scene.num_envs):
                image_path = args.output_dir / f"env_{env_id:02d}.png"
                save_images_to_file(images[env_id : env_id + 1].float() / 255.0, str(image_path))
                print(f"Env {env_id:02d}: right wrist actual X {actual_x[env_id].item():.4f} m; saved {image_path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
