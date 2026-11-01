# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from g1_real2sim.scenes.tabletop_scene_cfg import TabletopSceneCfg

from ... import mdp


@configclass
class ActionsCfg:
    """Define action terms for your robot."""

    action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Define observation groups and terms."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation terms."""

        observation = ObsTerm(func=mdp.observation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Define reset and other event terms."""

    reset = EventTerm(func=mdp.reset, mode="reset")


@configclass
class RewardsCfg:
    """Define task reward terms and weights."""

    reward = RewTerm(func=mdp.reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Define termination and timeout terms."""

    termination = DoneTerm(func=mdp.termination)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configure the manager-based task."""

    sim: SimulationCfg = SimulationCfg()
    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Set simulation timing, physics, and task parameters."""
        raise NotImplementedError("Configure simulation, scene, and manager terms.")
