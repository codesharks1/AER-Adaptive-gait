# Checkpoints

This directory contains the three checkpoints used by the released training pipeline.

| Checkpoint | Network | Observations | Description |
| --- | --- | --- | --- |
| `flat_pretrain.pt` | Feed-forward PPO ActorCritic | 247 | Flat-terrain policy used to initialize rough-terrain training. |
| `rough_terrain_teacher.pt` | Feed-forward PPO ActorCritic | 247 | Privileged teacher trained with rough-terrain curriculum. |
| `distilled_student.pt` | StudentTeacherRecurrent | student 57, teacher 247 | GRU student distilled from the rough-terrain teacher. |

The student checkpoint contains both teacher and student modules because it is the native RSL-RL distillation checkpoint. For deployment, use the student branch (`memory_s` and `student`).

## Provenance

| Published file | Original run | Original checkpoint |
| --- | --- | --- |
| `flat_pretrain.pt` | `2026-06-11_13-12-36` | `model_1999.pt` |
| `rough_terrain_teacher.pt` | `2026-06-14_19-01-45` | `model_4998.pt` |
| `distilled_student.pt` | `2026-06-15_16-18-47` | `model_2999.pt` |

Each run's saved environment and agent configuration is included in `training_configs/<stage>/`.

## SHA-256

Hashes are generated during publication and recorded in `SHA256SUMS.txt`.
