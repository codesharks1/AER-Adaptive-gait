"""Minimal IsaacLab-to-MuJoCo sim2sim runner for the exported Go2 student policy.

This is intentionally a single-thread scratchpad:
- no Unitree SDK
- no DDS
- fixed velocity command
- TorchScript recurrent policy.pt
- PD joint-position control written directly to ``mj_data.ctrl``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from load_go2_flat import DEFAULT_SCENE_PATH, find_unitree_go2_dir, prepare_scene_for_mujoco, print_model_summary


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "distilled_policy.pt"

MUJOCO_QPOS_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
MUJOCO_CTRL_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ISAAC_POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
POLICY_JOINT_ORDERS = {
    "isaac": ISAAC_POLICY_JOINT_NAMES,
    "qpos": MUJOCO_QPOS_JOINT_NAMES,
    "unitree_ctrl": MUJOCO_CTRL_JOINT_NAMES,
}
DEFAULT_Q_BY_NAME = {
    "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.5,
    "RR_hip_joint": 0.0, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.5,
}
ACTION_SCALE_BY_SUFFIX = {
    "hip_joint": 0.125,
    "thigh_joint": 0.25,
    "calf_joint": 0.25,
}
KP = 25.0
KD = 0.5
TORQUE_LIMIT = 23.5
COMMAND_ARROW_COLOR = np.array([0.1, 0.35, 1.0, 1.0], dtype=np.float32)
ACTUAL_ARROW_COLOR = np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32)


def action_scale_for_joint(joint_name: str) -> float:
    for suffix, scale in ACTION_SCALE_BY_SUFFIX.items():
        if joint_name.endswith(suffix):
            return scale
    raise KeyError(f"No action scale configured for joint: {joint_name}")


def indices_for_order(source_names: list[str], target_names: list[str]) -> np.ndarray:
    return np.array([source_names.index(name) for name in target_names], dtype=np.int64)


def build_joint_order(policy_order_name: str) -> dict[str, np.ndarray | list[str]]:
    policy_names = POLICY_JOINT_ORDERS[policy_order_name]
    qpos_to_policy = indices_for_order(MUJOCO_QPOS_JOINT_NAMES, policy_names)
    ctrl_to_policy = indices_for_order(policy_names, MUJOCO_CTRL_JOINT_NAMES)
    default_q_policy = np.array([DEFAULT_Q_BY_NAME[name] for name in policy_names], dtype=np.float32)
    action_scale_policy = np.array([action_scale_for_joint(name) for name in policy_names], dtype=np.float32)
    return {
        "policy_names": policy_names,
        "qpos_to_policy": qpos_to_policy,
        "ctrl_to_policy": ctrl_to_policy,
        "default_q_policy": default_q_policy,
        "action_scale_policy": action_scale_policy,
        "kp_policy": np.full(12, KP, dtype=np.float32),
        "kd_policy": np.full(12, KD, dtype=np.float32),
        "torque_limit_policy": np.full(12, TORQUE_LIMIT, dtype=np.float32),
    }


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame using a wxyz quaternion."""
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(quat_conjugate(q_wxyz), v_quat), q_wxyz)[1:]


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into the world frame using a wxyz quaternion."""
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(q_wxyz, v_quat), quat_conjugate(q_wxyz))[1:]


def add_viewer_arrow(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    color: np.ndarray,
    width: float = 0.025,
) -> None:
    """Append a non-physical arrow to the MuJoCo user scene."""
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
    scene.ngeom += 1


def update_velocity_arrows(viewer, data: mujoco.MjData, command: np.ndarray, scale: float) -> None:
    """Draw commanded (blue) and measured (green) planar velocity arrows."""
    scene = viewer.user_scn
    scene.ngeom = 0

    base_pos = np.asarray(data.qpos[0:3], dtype=np.float64)
    base_quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float32)
    command_body = np.array([command[0], command[1], 0.0], dtype=np.float32)
    command_world = quat_rotate(base_quat_wxyz, command_body).astype(np.float64)
    actual_world = np.array([data.qvel[0], data.qvel[1], 0.0], dtype=np.float64)

    command_start = base_pos + np.array([0.0, 0.0, 0.38])
    actual_start = base_pos + np.array([0.0, 0.0, 0.31])
    if np.linalg.norm(command_world[:2]) > 1.0e-6:
        add_viewer_arrow(scene, command_start, command_start + scale * command_world, COMMAND_ARROW_COLOR)
    if np.linalg.norm(actual_world[:2]) > 1.0e-6:
        add_viewer_arrow(scene, actual_start, actual_start + scale * actual_world, ACTUAL_ARROW_COLOR)


def reset_robot_to_training_stance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.38]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:19] = np.array([DEFAULT_Q_BY_NAME[name] for name in MUJOCO_QPOS_JOINT_NAMES], dtype=np.float32)
    mujoco.mj_forward(model, data)


def make_student_obs(
    data: mujoco.MjData,
    command: np.ndarray,
    last_action_policy: np.ndarray,
    last_torque_policy: np.ndarray,
    joint_order: dict[str, np.ndarray | list[str]],
) -> np.ndarray:
    """Construct the 57-dimensional deployable student observation."""
    base_ang_vel = np.array(data.sensor("imu_gyro").data, dtype=np.float32) * 0.2

    base_quat_wxyz = np.array(data.qpos[3:7], dtype=np.float32)
    projected_gravity = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    # 索引不一样是因为有四元数
    q_policy = np.array(data.qpos[7:19], dtype=np.float32)[joint_order["qpos_to_policy"]]
    qd_policy = np.array(data.qvel[6:18], dtype=np.float32)[joint_order["qpos_to_policy"]]
    joint_pos_rel = q_policy - joint_order["default_q_policy"]
    joint_vel_rel = qd_policy * 0.05
    joint_effort = last_torque_policy * 0.01

    obs = np.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            command.astype(np.float32),
            joint_pos_rel,
            joint_vel_rel,
            joint_effort,
            last_action_policy,
        ]
    ).astype(np.float32)

    if obs.shape != (57,):
        raise RuntimeError(f"Expected 57-dim student obs, got {obs.shape}")
    return obs


def apply_pd_control(
    data: mujoco.MjData,
    action_policy: np.ndarray,
    action_clip: float,
    joint_order: dict[str, np.ndarray | list[str]],
) -> np.ndarray:
    action_policy = np.clip(action_policy, -action_clip, action_clip)
    target_q_policy = joint_order["default_q_policy"] + joint_order["action_scale_policy"] * action_policy

    q_policy = np.array(data.qpos[7:19], dtype=np.float32)[joint_order["qpos_to_policy"]]
    qd_policy = np.array(data.qvel[6:18], dtype=np.float32)[joint_order["qpos_to_policy"]]

    ctrl_to_policy = joint_order["ctrl_to_policy"]
    target_q_ctrl = target_q_policy[ctrl_to_policy]
    q_ctrl = q_policy[ctrl_to_policy]
    qd_ctrl = qd_policy[ctrl_to_policy]
    kp_ctrl = joint_order["kp_policy"][ctrl_to_policy]
    kd_ctrl = joint_order["kd_policy"][ctrl_to_policy]
    limit_ctrl = joint_order["torque_limit_policy"][ctrl_to_policy]

    torque_ctrl = kp_ctrl * (target_q_ctrl - q_ctrl) + kd_ctrl * (0.0 - qd_ctrl)
    torque_ctrl = np.clip(torque_ctrl, -limit_ctrl, limit_ctrl)
    data.ctrl[:] = torque_ctrl

    torque_policy = np.zeros(12, dtype=np.float32)
    for ctrl_index, policy_index in enumerate(ctrl_to_policy):
        torque_policy[policy_index] = torque_ctrl[ctrl_index]
    return torque_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exported Go2 student policy in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--unitree-mujoco-dir", type=Path, default=None)
    parser.add_argument("--policy-joint-order", choices=sorted(POLICY_JOINT_ORDERS), default="isaac")
    parser.add_argument("--command-x", type=float, default=0.2)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--action-clip", type=float, default=2.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--hold-only", action="store_true", help="Ignore policy and hold the default standing pose with PD.")
    parser.add_argument("--debug", action="store_true", help="Print one-line diagnostics every 0.5 seconds.")
    parser.add_argument("--obs-debug", action="store_true", help="Print segmented student observations before policy inference.")
    parser.add_argument("--velocity-arrow-scale", type=float, default=1.0, help="Scale applied to viewer velocity arrows.")
    parser.add_argument("--hide-velocity-arrows", action="store_true", help="Hide commanded and measured velocity arrows.")
    args = parser.parse_args()
    # 重新解析路径，确保它们是绝对路径
    scene_path = args.scene.resolve()
    policy_path = args.policy.resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Cannot find exported policy: {policy_path}")
    # 解析关节顺序
    joint_order = build_joint_order(args.policy_joint_order)
    print(f"[INFO] Policy joint order preset: {args.policy_joint_order}")
    for index, name in enumerate(joint_order["policy_names"]):
        print(f"  policy[{index:02d}] = {name}")
    # 找到 unitree_mujoco 仓库目录，准备运行时的 MuJoCo 场景 XML
    go2_dir = find_unitree_go2_dir(args.unitree_mujoco_dir)
    runtime_scene_path = prepare_scene_for_mujoco(scene_path, go2_dir)
    # 加载 MuJoCo 模型和数据
    model = mujoco.MjModel.from_xml_path(str(runtime_scene_path))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    # 重置机器人到训练姿态
    reset_robot_to_training_stance(model, data)
    # 输出MuJoCo模型的关节和执行器信息
    print_model_summary(model)
    # 加载 TorchScript 导出的策略
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    # 清空GRU的隐藏状态
    if hasattr(policy, "hidden_state"):
        policy.hidden_state.zero_()
    print(f"[INFO] Loaded policy: {policy_path}")

    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    last_action_policy = np.zeros(12, dtype=np.float32)
    last_torque_policy = np.zeros(12, dtype=np.float32)
    policy_steps = 0
    # 用于debug测试用的
    def print_obs_debug(obs: np.ndarray, action: np.ndarray | None = None) -> None:
        segments = [
            ("base_ang_vel*0.2", obs[0:3]),
            ("projected_gravity", obs[3:6]),
            ("command", obs[6:9]),
            ("joint_pos_rel", obs[9:21]),
            ("joint_vel_rel*0.05", obs[21:33]),
            ("joint_effort*0.01", obs[33:45]),
            ("last_action", obs[45:57]),
        ]
        print("[OBS DEBUG]")
        for name, values in segments:
            print(
                f"  {name:18s} min={values.min(): .4f} max={values.max(): .4f} "
                f"mean={values.mean(): .4f} values={np.array2string(values, precision=3, suppress_small=True)}"
            )
        if action is not None:
            print(f"  action raw          {np.array2string(action, precision=3, suppress_small=True)}")

    def step_policy_and_physics() -> None:
        # 当前变量不是内部函数，需要更新他
        nonlocal last_action_policy, last_torque_policy, policy_steps
        if args.hold_only:
            last_action_policy = np.zeros(12, dtype=np.float32)
        else:
            obs = make_student_obs(data, command, last_action_policy, last_torque_policy, joint_order)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy()[0].astype(np.float32)
            if args.obs_debug and policy_steps < 5:
                print_obs_debug(obs, action)
            last_action_policy = np.clip(action, -args.action_clip, args.action_clip)
        for _ in range(args.decimation):
            last_torque_policy = apply_pd_control(data, last_action_policy, args.action_clip, joint_order)
            mujoco.mj_step(model, data)
        policy_steps += 1
        if args.debug and policy_steps % max(1, int(0.5 / (args.sim_dt * args.decimation))) == 0:
            print(
                f"t={data.time:6.3f} z={data.qpos[2]: .3f} "
                f"quat={data.qpos[3:7]} "
                f"action_abs_max={np.max(np.abs(last_action_policy)): .3f} "
                f"torque_abs_max={np.max(np.abs(last_torque_policy)): .3f}"
            )

    if args.no_viewer:
        num_policy_steps = int(args.duration / (args.sim_dt * args.decimation))
        for _ in range(num_policy_steps):
            step_policy_and_physics()
        print(f"[INFO] Headless policy run finished at t={data.time:.3f}s")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < args.duration:
            wall_start = time.time()
            step_policy_and_physics()
            if not args.hide_velocity_arrows:
                with viewer.lock():
                    update_velocity_arrows(viewer, data, command, args.velocity_arrow_scale)
            viewer.sync()
            policy_dt = args.sim_dt * args.decimation
            sleep_time = policy_dt - (time.time() - wall_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
