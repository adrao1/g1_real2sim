"""Print the G1 wrist poses immediately after an environment reset."""

from isaaclab.app import launch_simulation

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    cfg = PickPlaceEnvCfg()
    with launch_simulation(cfg, {"device": "cpu"}):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            poses = robot.data.body_link_pose_w.torch[0]
            origin = env.scene.env_origins[0]

            print(f"Environment origin: {origin.tolist()}")
            for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
                body_id = robot.body_names.index(name)
                pose = poses[body_id]
                print(f"{name} world position (m): {pose[:3].tolist()}")
                print(f"{name} action position (m): {(pose[:3] - origin).tolist()}")
                print(f"{name} rotation (xyzw): {pose[3:7].tolist()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
