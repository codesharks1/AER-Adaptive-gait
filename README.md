# AER Adaptive Gait for Unitree Go2

Isaac Lab implementation of adaptive, energy-regularized quadruped locomotion, including rough-terrain teacher training and recurrent teacher-student policy distillation.

The released pipeline contains three stages:

1. Train a feed-forward PPO policy on flat terrain.
2. Continue training the same policy as a privileged teacher on curriculum-generated rough terrain.
3. Distill the teacher into a recurrent student that does not observe base linear velocity or the terrain height scanner.

## Policy Structure

The teacher receives a 247-dimensional observation containing proprioception, base linear velocity, and a 187-dimensional height scan. The deployable student receives 57-dimensional proprioception and uses a GRU to infer hidden terrain and velocity information from observation history:

```text
Teacher: 247 observations -> MLP [512, 256, 128] -> 12 actions
Student:  57 observations -> GRU(247) -> MLP [512, 256, 128] -> 12 actions
```

Distillation minimizes a Huber imitation loss between the teacher and student actions. An optional recurrent PPO stage can fine-tune the distilled student using environment rewards.

## Installation

Install Isaac Lab and `rsl-rl-lib>=3.0.1`, then install this extension from the repository root:

```bash
python -m pip install -e source/go2_demo
python scripts/list_envs.py
```

The repository includes the local Unitree Go2 USD files used by `unitree.py`; no machine-specific model path is required.

## Released Checkpoints

| File | Role | Training source |
| --- | --- | --- |
| `checkpoints/flat_pretrain.pt` | PPO flat-terrain initialization | `2026-06-11_13-12-36/model_1999.pt` |
| `checkpoints/rough_terrain_teacher.pt` | PPO privileged rough-terrain teacher | `2026-06-14_19-01-45/model_4998.pt` |
| `checkpoints/distilled_student.pt` | Recurrent teacher-student distillation result | `2026-06-15_16-18-47/model_2999.pt` |

The original `agent.yaml` and `env.yaml` files are stored under `checkpoints/training_configs/`.

## Play

Play the rough-terrain teacher:

```bash
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --num_envs 40 --checkpoint checkpoints/rough_terrain_teacher.pt
```

Play the distilled recurrent student:

```bash
python scripts/rsl_rl/play.py --task Go2-velocity-Distill-v0 --num_envs 40 --checkpoint checkpoints/distilled_student.pt
```

Add `--video --video_length 500` to either command to record a rollout.

## Train

Train the PPO teacher task from scratch:

```bash
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 400 --max_iterations 5000 --headless --video
```

Distill the released teacher:

```bash
python scripts/rsl_rl/train.py --task Go2-velocity-Distill-v0 --num_envs 400 --max_iterations 3000 --checkpoint checkpoints/rough_terrain_teacher.pt --headless --video
```

Fine-tune the released student with recurrent PPO:

```bash
python scripts/rsl_rl/train.py --task Go2-velocity-StudentFinetune-v0 --num_envs 400 --max_iterations 1000 --resume --checkpoint checkpoints/distilled_student.pt --headless --video
```

The fine-tuning entry point maps the distillation checkpoint keys `memory_s` and `student` to the recurrent PPO keys `memory_a` and `actor`. The PPO critic is initialized separately and learned from rewards.

## Main Files

- `go2_demo_velocity.py`: terrain generator, observations, rewards, commands, events, and curriculum configuration.
- `aer_env.py`: adaptive energy reward aggregation.
- `agents/rsl_rl_ppo_cfg.py`: PPO, distillation, and recurrent student fine-tuning configurations.
- `scripts/rsl_rl/train.py`: runner selection and distilled-student checkpoint conversion for PPO fine-tuning.
- `checkpoints/README.md`: checkpoint architecture, provenance, and integrity hashes.

## Notes

- The rough-terrain teacher was initialized from the released flat-terrain checkpoint.
- The student excludes `base_lin_vel` and `height_scanner`; its GRU uses observation history to estimate the missing information implicitly.
- Training remains stochastic even with a fixed seed because GPU simulation and optimization can introduce nondeterminism.
