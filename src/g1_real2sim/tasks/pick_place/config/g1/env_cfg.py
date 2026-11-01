# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    JointPositionActionCfg,
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
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from g1_real2sim.scenes.tabletop_scene_cfg import TabletopSceneCfg


@configclass
class ActionsCfg:
    """Control G1 joints relative to their default positions."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


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
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2
        self.sim.physics = PhysxCfg()
        self.sim.visualizer_cfgs = VisualizerCfg(
            eye=(2.5, 2.5, 1.8),
            lookat=(0.0, 0.0, 0.8),
        )
