# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    joint_pos_rel,
    joint_vel_rel,
    reset_scene_to_default,
    time_out,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import clone, configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.contrib.locomanip_pick_place.configs.pink_controller_cfg import G1_UPPER_BODY_IK_ACTION_CFG

from g1_real2sim.scenes.tabletop_scene_cfg import TabletopSceneCfg


@configclass
class ActionsCfg:
    """Control both G1 wrists with Pink IK and the hands with joint targets."""

    upper_body_ik = clone(G1_UPPER_BODY_IK_ACTION_CFG)


@configclass
class ObservationsCfg:
    """Define observation groups and terms."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation terms."""

        joint_pos = ObsTerm(func=joint_pos_rel)
        joint_vel = ObsTerm(func=joint_vel_rel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Define reset and other event terms."""

    reset = EventTerm(func=reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """End each episode when its time limit is reached."""

    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configure the manager-based task."""

    sim: SimulationCfg = SimulationCfg()
    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Configure simulation timing and the Isaac Sim PhysX backend."""
        # Generate the IK model from this scene's USD. Its exported URDF uses
        # unprefixed link names, unlike the prebuilt URDF in the Isaac Lab example.
        controller = self.actions.upper_body_ik.controller
        controller.usd_path = self.scene.robot.spawn.usd_path
        left_wrist_task, right_wrist_task, posture_task = controller.variable_input_tasks
        left_wrist_task.frame = "left_wrist_yaw_link"
        right_wrist_task.frame = "right_wrist_yaw_link"
        left_wrist_task.base_link_frame_name = "pelvis"
        right_wrist_task.base_link_frame_name = "pelvis"
        posture_task.controlled_frames = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2
        self.sim.physics = PhysxCfg()
        self.sim.visualizer_cfgs = VisualizerCfg(
            eye=(2.5, 2.5, 1.8),
            lookat=(0.0, 0.0, 0.8),
        )
