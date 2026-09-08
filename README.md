# AER Adaptive Gait for Unitree Go2

Isaac Lab implementation of adaptive energy-regularized quadruped locomotion, recurrent teacher-student distillation, symmetry-augmented training, and MuJoCo Sim2Sim deployment.

The current release follows this pipeline:

1. Train a feed-forward PPO policy on flat terrain.
2. Continue training the privileged teacher on curriculum-generated rough terrain.
3. Use left-right symmetry data augmentation during PPO training to reduce asymmetric gait artifacts.
4. Distill the 247-dimensional privileged teacher into a recurrent 57-dimensional student.
5. Export the student and validate it in MuJoCo on flat ground and stairs.

## Policy Structure

```text
Teacher: 247 observations -> MLP [512, 256, 128] -> 12 actions
Student:  57 observations -> GRU(57, 247) -> MLP [512, 256, 128] -> 12 actions
```

The teacher observes proprioception, base linear velocity, and a 187-dimensional terrain height scan. The deployable student removes `base_lin_vel` and `height_scanner`; its GRU uses observation history to infer the missing information. Distillation minimizes a Huber loss between teacher and student actions.

## Compatibility

The released code and checkpoints were developed with:

- Isaac Lab commit `3206bca216c220216e46a2974be2cfc4b48761a4`
- `rsl-rl-lib==3.1.2`
- Python 3.11

A compatible Isaac Lab checkout can be prepared with:

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab-AER
cd IsaacLab-AER
git checkout 3206bca216c220216e46a2974be2cfc4b48761a4
.\isaaclab.bat --install
pip install --force-reinstall rsl-rl-lib==3.1.2
```

## Installation

From this repository root:

```powershell
python -m pip install -e source/go2_demo
python scripts/list_envs.py
```

The repository contains the local Go2 USD assets required by the relative asset configuration.

## Released Checkpoints

| File | Role | Training source |
| --- | --- | --- |
| `checkpoints/flat_pretrain.pt` | Symmetry-augmented flat-terrain PPO initialization | `2026-07-28_21-31-50/model_2998.pt` |
| `checkpoints/rough_terrain_teacher.pt` | Privileged rough-terrain PPO teacher | `2026-07-29_17-03-20/model_6997.pt` |
| `checkpoints/distilled_student.pt` | Recurrent teacher-student distillation checkpoint | `2026-07-29_19-44-17/model_2999.pt` |
| `checkpoints/distilled_policy.pt` | Exported TorchScript student for MuJoCo | `2026-07-29_19-44-17/exported/policy.pt` |
| [`checkpoints/distilled_policy.onnx`](checkpoints/distilled_policy.onnx) | Exported ONNX recurrent student | `2026-07-29_19-44-17/exported/policy.onnx` |

The matching `agent.yaml` and `env.yaml` files are stored under `checkpoints/training_configs/`.

The ONNX export uses explicit GRU hidden-state inputs and outputs. See the [ONNX interface and example](checkpoints/README.md#onnx-inference) before using it. The current `sim2sim/run_policy_flat.py` loads TorchScript `.pt` files, not ONNX files.

## Play in Isaac Lab

Play the rough-terrain teacher:

```powershell
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --num_envs 40 --checkpoint checkpoints/rough_terrain_teacher.pt
```

Play the recurrent student:

```powershell
python scripts/rsl_rl/play.py --task Go2-velocity-Distill-v0 --num_envs 40 --checkpoint checkpoints/distilled_student.pt
```

Add `--video --video_length 500` to record a rollout.

## Train

Train the PPO teacher from scratch:

```powershell
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 400 --max_iterations 5000 --headless --video
```

Continue rough-terrain training from the released flat checkpoint:

```powershell
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 400 --max_iterations 7000 --resume --checkpoint checkpoints/flat_pretrain.pt --headless --video
```

Distill the released teacher:

```powershell
python scripts/rsl_rl/train.py --task Go2-velocity-Distill-v0 --num_envs 400 --max_iterations 3000 --resume --checkpoint checkpoints/rough_terrain_teacher.pt --headless --video
```

Fine-tune the student with recurrent PPO:

```powershell
python scripts/rsl_rl/train.py --task Go2-velocity-StudentFinetune-v0 --num_envs 400 --max_iterations 500 --resume --checkpoint checkpoints/distilled_student.pt --headless --video
```

Fine-tuning converts `memory_s.*` to `memory_a.*` and `student.*` to `actor.*`. The PPO critic is initialized separately. Use a conservative learning rate because aggressive PPO updates can damage the distilled gait.

## MuJoCo Sim2Sim

Install MuJoCo and clone Unitree's model repository beside this repository:

```powershell
pip install mujoco
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

Run the minimal MuJoCo example:

```powershell
python sim2sim/mujoco_minmal/mujoco_min.py
```

Load the Go2 scene without a policy:

```powershell
python sim2sim/load_go2_flat.py
```

Run the exported student on flat ground:

```powershell
python sim2sim/run_policy_flat.py --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-y 0 --command-yaw 0 --duration 30 --debug --action-clip 10
```

Run the same policy on stairs:

```powershell
python sim2sim/run_policy_flat.py --scene sim2sim/scenes/go2_stairs_scene.xml --policy checkpoints/distilled_policy.pt --command-x 0.3 --command-y 0 --command-yaw 0 --duration 30 --debug --action-clip 10
```

See [`sim2sim/README.md`](sim2sim/README.md) for joint ordering, scene discovery, velocity arrows, URDF conversion, and additional commands.

## Main Files

- `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/go2_demo_velocity.py`: terrain, observations, rewards, commands, events, and curricula.
- `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/mdp/symmetry.py`: Go2 left-right observation and action mirroring.
- `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/aer_env.py`: adaptive energy reward aggregation.
- `source/go2_demo/go2_demo/tasks/manager_based/go2_demo/agents/rsl_rl_ppo_cfg.py`: PPO, distillation, and recurrent fine-tuning configurations.
- `scripts/rsl_rl/train.py`: runner selection and distilled checkpoint adaptation.
- `sim2sim/run_policy_flat.py`: single-process MuJoCo policy loop and PD control.
- `sim2sim/scenes/`: flat-ground and stair scenes.
- `sim2sim/urdf_to_mjcf/convert.py`: URDF to MJCF conversion utility.

## Notes

- The policy command is expressed in the robot body frame. A zero yaw-rate command is not a global heading controller.
- Sim2Sim depends on consistent observation scaling, joint ordering, action scaling, PD gains, control frequency, and coordinate conventions.
- Training remains stochastic even with a fixed seed because GPU simulation and optimization can introduce nondeterminism.
