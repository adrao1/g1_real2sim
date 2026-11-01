"""Fixed-base G1 tabletop scene with a graspable object."""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, GroundPlaneCfg, UsdFileCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils import clone, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from g1_real2sim.assets import G1_REAL2SIM_ASSETS_DIR

UNITREE_G1_USD = (
    G1_REAL2SIM_ASSETS_DIR
    / "robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Scene containing a fixed-base G1 robot and a tabletop cracker box."""

    robot: ArticulationCfg = clone(G1_29DOF_CFG)

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0.40, 0.87),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=UsdPhysicsRigidBodyCfg(),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    def __post_init__(self) -> None:
        self.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.robot.spawn.usd_path = str(UNITREE_G1_USD)
        self.robot.spawn.fix_root_link = True
