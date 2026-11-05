"""Move G1 wrists to grasp poses and save one image per grasp."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evaluate_contact_grasps import DEFAULT_GRASPS, configure_item, configure_tracked_item, load_grasps

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--output-dir", type=Path, help="Image directory; defaults to a new directory for each run")
    parser.add_argument("--tracking-config", type=Path, help="Place the item at the recording's first world pose")
    parser.add_argument("--grasp-file", type=Path, help="Isaac grasp YAML; defaults to the item's grasp file")
    parser.add_argument("--grasps", nargs="+", type=int, metavar="ID", help="Grasp IDs to visualize; default is all")
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    cfg = PickPlaceEnvCfg()
    try:
        if args.tracking_config is None:
            item = configure_item(cfg, argparse.Namespace(item_config=None, item_position=None, item_rotation=None))
        else:
            item, first_frame = configure_tracked_item(cfg, args.tracking_config)
            print(f"Tracking frame {first_frame}; item pose: {item.position}, {item.rotation_xyzw}", flush=True)
        grasp_file = args.grasp_file or item.grasp_file or (DEFAULT_GRASPS if item.name == "cracker_box" else None)
        if grasp_file is None:
            raise ValueError(f"Provide --grasp-file for {item.name}")
        grasp_file = grasp_file.expanduser().resolve(strict=True)
        grasps = load_grasps(grasp_file)
        grasp_ids = args.grasps if args.grasps is not None else list(grasps)
        if not grasp_ids or len(set(grasp_ids)) != len(grasp_ids) or any(index not in grasps for index in grasp_ids):
            raise ValueError("Select distinct grasp IDs present in the selected grasp YAML")
        if args.output_dir is None:
            source = args.tracking_config.stem if args.tracking_config is not None else f"{item.name}_preset"
            run_name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
            args.output_dir = Path("outputs/grasp_visualization") / source / run_name
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Grasp images: {args.output_dir}", flush=True)

    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = len(grasp_ids)
    cfg.scene.object.spawn.collision_props = UsdPhysicsCollisionCfg(collision_enabled=False)
    cfg.scene.object.spawn.rigid_props = [
        UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
        PhysxRigidBodyCfg(disable_gravity=True),
    ]
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
        from isaaclab.utils.math import (
            combine_frame_transforms,
            quat_error_magnitude,
            quat_from_euler_xyz,
            subtract_frame_transforms,
        )

        print("Converting G1 USD to URDF for IK...", flush=True)
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
            focus = torch.tensor(item.position, device=env.device) + torch.tensor((0.0, 0.0, -0.02), device=env.device)
            camera.set_world_poses_from_view(
                origins + focus + torch.tensor((1.2, 0.8, 0.65), device=env.device),
                origins + focus,
            )
            wrist_ids = [robot.body_names.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
            wrist_poses = robot.data.body_link_pose_w.torch[:, wrist_ids].clone()
            wrist_poses[..., :3] -= origins[:, None, :]
            hand_ids, _ = robot.find_joints(cfg.actions.upper_body_ik.hand_joint_names)

            action = torch.zeros_like(env.action_manager.action)
            action[:, :14] = wrist_poses.flatten(start_dim=1)
            action[:, 14:] = robot.data.joint_pos.torch[:, hand_ids]

            object_pose = env.scene["object"].data.root_link_pose_w.torch
            grasp_pos = torch.tensor(
                [grasps[index]["position"] for index in grasp_ids], device=env.device, dtype=object_pose.dtype
            )
            grasp_quat = torch.tensor(
                [grasps[index]["quaternion_xyzw"] for index in grasp_ids],
                device=env.device,
                dtype=object_pose.dtype,
            )
            grasp_pos_w, grasp_quat_w = combine_frame_transforms(
                object_pose[:, :3], object_pose[:, 3:7], grasp_pos, grasp_quat
            )

            # GraspGenX unitree_g1/gripper.urdf: world_joint -> right_palm_link.
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

            # Assume the GraspGenX palm frame matches the simulator palm frame.
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

            with torch.inference_mode():
                for step in range(100):
                    fraction = (step + 1) / 100
                    action[:, 7:10] = start_pos + fraction * (target_pos - start_pos)
                    slerp_quat = (
                        torch.sin((1 - fraction) * angle) * start_quat + torch.sin(fraction * angle) * end_quat
                    ) / sin_angle.clamp_min(1e-6)
                    lerp_quat = torch.nn.functional.normalize((1 - fraction) * start_quat + fraction * end_quat, dim=-1)
                    action[:, 10:14] = torch.where(sin_angle > 1e-6, slerp_quat, lerp_quat)
                    env.step(action)
                    if (step + 1) % 25 == 0:
                        print(f"Move step {step + 1}/100", flush=True)
                print("Holding final pose for 100 steps...", flush=True)
                for _ in range(100):
                    env.step(action)

            actual_pose = robot.data.body_link_pose_w.torch[:, wrist_ids[1]]
            position_errors = torch.linalg.norm(actual_pose[:, :3] - target_pos_w, dim=-1)
            rotation_errors = torch.rad2deg(quat_error_magnitude(actual_pose[:, 3:7], target_quat))
            print("Capturing camera images...", flush=True)
            images = camera.data.output["rgb"].torch
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for slot, grasp_id in enumerate(grasp_ids):
                image_path = args.output_dir / f"grasp_{grasp_id}.png"
                save_images_to_file(images[slot : slot + 1].float() / 255.0, str(image_path))
                print(
                    f"grasp_{grasp_id} wrist error: {position_errors[slot].item():.4f} m, "
                    f"{rotation_errors[slot].item():.1f} deg; saved {image_path}"
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
