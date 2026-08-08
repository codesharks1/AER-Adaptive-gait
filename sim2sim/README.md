# Go2 MuJoCo Sim2Sim

This directory contains a minimal, single-process MuJoCo deployment path for the recurrent Go2 student policy.

## Contents

- `mujoco_minmal/`: official-style minimal MuJoCo XML and Python loop.
- `urdf_to_mjcf/convert.py`: converts a tree-structured URDF into MJCF.
- `load_go2_flat.py`: loads the Unitree Go2 MJCF into a flat scene.
- `run_policy_flat.py`: observation construction, GRU inference, action mapping, PD control, and physics stepping.
- `scenes/go2_flat_scene.xml`: flat ground.
- `scenes/go2_stairs_scene.xml`: stair test scene.

## Setup

```powershell
pip install mujoco
```

Clone Unitree's MuJoCo repository next to the project repository, or pass its path explicitly:

```powershell
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

The scripts search common sibling and nested layouts. Override discovery when needed:

```powershell
python sim2sim/load_go2_flat.py --unitree-mujoco-dir D:\path\to\unitree_mujoco
```

## Minimal Examples

```powershell
python sim2sim/mujoco_minmal/mujoco_min.py
python sim2sim/load_go2_flat.py
python sim2sim/load_go2_flat.py --no-viewer --duration 2
```

Convert a URDF to MJCF:

```powershell
python sim2sim/urdf_to_mjcf/convert.py --urdf path\to\robot.urdf --output robot.xml
```

The generated MJCF is a starting point. Actuators, sensors, mesh paths, inertial properties, collision geometry, and control limits still need to be checked.

## Run the Student Policy

From the repository root, run flat ground:

```powershell
python sim2sim/run_policy_flat.py --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-y 0 --command-yaw 0 --duration 30 --debug --action-clip 10
```

Run stairs:

```powershell
python sim2sim/run_policy_flat.py --scene sim2sim/scenes/go2_stairs_scene.xml --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-y 0 --command-yaw 0 --duration 30 --debug --action-clip 10
```

Turn left or right by changing yaw command:

```powershell
python sim2sim/run_policy_flat.py --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-yaw 0.3 --duration 30 --action-clip 10
python sim2sim/run_policy_flat.py --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-yaw -0.3 --duration 30 --action-clip 10
```

The viewer shows commanded planar velocity with a blue arrow and measured planar velocity with a green arrow. Use `--hide-velocity-arrows` to disable them.

## Interface Alignment

The policy uses Isaac Lab joint order while Unitree's MuJoCo actuators use a different order. `run_policy_flat.py` explicitly maps:

- MuJoCo `qpos/qvel` into policy observation order.
- Policy target joint positions into MuJoCo actuator order.
- MuJoCo torques back into policy order for the next observation.

The deployment loop also preserves the training action scale, default joint offsets, PD gains, torque limits, simulation timestep, policy decimation, and recurrent hidden state.