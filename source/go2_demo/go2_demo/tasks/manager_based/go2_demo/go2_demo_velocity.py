import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_demo.assets.robot.unitree import UNITREE_GO2_CFG as RobotCFG
from go2_demo.tasks.manager_based.go2_demo import mdp
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # 超平坦地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = RobotCFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 高度传感
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 机体根部线速度（坐标系：asset/root frame，也就是机器人根坐标系，不是世界坐标系）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=1.0)
        # 机体根部角速度（坐标系：asset/root frame）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        # 重力方向在机器人根坐标系下的投影（坐标系：asset/root frame）
        # 常用来表征机器人当前姿态，比如有没有侧倾、俯仰
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # 速度指令（由 command manager 生成）
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        # 相对关节位置（相对默认关节位置的偏差）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        # 相对关节速度（相对默认关节速度的偏差）
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 关节输出力 / 力矩（单位通常是 N 或 N·m）
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        # 上一时刻的动作输出
        # 一般就是上一拍策略网络送给环境的 action，可用于提升控制平滑性或提供时序信息
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # 高度扫描结果（坐标系：sensor frame，也就是 height_scanner 这个传感器自己的坐标系）
        # 本质上是传感器射线打到地面后得到的局部高度信息
        # 这里的 sensor_cfg 指向你在 scene 里定义好的 "height_scanner"
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
    critic: CriticCfg = CriticCfg()
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 6.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 6.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )

@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.001)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # 惩罚关节输出力矩过大
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # 惩罚关节速度
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    # 惩罚关节加速度
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚相邻时刻动作变化过大
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-6.25e-3)
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # 限制那些本来不应该着地的身体部位不要乱碰?
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )
    

@configclass
class TerminationsCfg:
    # 训练时间到了
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机身碰撞地，此项需要场景类中配置接触力传感器
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 机器人倾斜太厉害，超过配置的阈值0.8 
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels
    )

@configclass
class GO2RobotDemoEnv(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
