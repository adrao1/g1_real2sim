"""Move the G1 right wrist 0.20 m along the environment's +X axis."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    cfg = PickPlaceEnvCfg()
    cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        import torch

        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
            wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
            wrist_poses[..., :3] -= env.scene.env_origins[:, None, :]
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
                for _ in range(100):
                    env.step(action)

            actual_x = robot.data.body_link_pose_w.torch[0, wrist_ids[1], 0] - env.scene.env_origins[0, 0]
            print(f"Right wrist actual X: {actual_x.item():.4f} m")
        finally:
            env.close()


if __name__ == "__main__":
    main()
