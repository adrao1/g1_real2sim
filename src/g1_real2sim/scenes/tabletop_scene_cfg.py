"""G1 scene configuration, starting with an anchored robot."""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.utils import clone, configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Scene containing a G1 robot with its base fixed in place."""

    robot: ArticulationCfg = clone(G1_29DOF_CFG)

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    def __post_init__(self) -> None:
        self.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.robot.spawn.fix_root_link = True
