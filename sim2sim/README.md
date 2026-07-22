# Go2 MuJoCo Sim2Sim Scratchpad

This folder starts with the smallest possible MuJoCo loop:

1. Load a flat scene XML from `scenes/go2_flat_scene.xml`.
2. Include Unitree's official Go2 MJCF.
3. Open MuJoCo's passive viewer.

## Setup

```cmd
pip install mujoco
```

Clone Unitree's MuJoCo repository next to this folder:

```cmd
cd /d D:\RobotProject
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

The loader auto-detects these layouts:

```text
parent_folder/
├─ simtosim/
└─ unitree_mujoco/
```

or:

```text
simtosim/
└─ unitree_mujoco/
```

You can also pass the path explicitly:

```cmd
python load_go2_flat.py --unitree-mujoco-dir D:\path\to\unitree_mujoco
```

## Run

```cmd
python load_go2_flat.py
```

Use a different scene:

```cmd
python load_go2_flat.py --scene scenes\go2_flat_scene.xml
```

Headless smoke test:

```cmd
python load_go2_flat.py --no-viewer --duration 2
```

## Next steps

- Add PD hold control.
- Read qpos/qvel/IMU and construct IsaacLab student observations.
- Load the fine-tuned recurrent student policy.
- Maintain GRU hidden state.
- Convert policy actions to target joint positions.

## Run the Exported Student Policy

Minimal single-thread sim2sim loop:

```cmd
python run_policy_flat.py --policy D:\RobotProject\go2_demo\logs\rsl_rl\go2_demo\2026-07-07_20-45-47\exported\policy.pt
```

Headless smoke test:

```cmd
python run_policy_flat.py --no-viewer --duration 2 --policy D:\RobotProject\go2_demo\logs\rsl_rl\go2_demo\2026-07-07_20-45-47\exported\policy.pt
```

The first version assumes IsaacLab policy joint order is `FL, FR, RL, RR` and maps actions into Unitree MuJoCo actuator order `FR, FL, RR, RL`.
