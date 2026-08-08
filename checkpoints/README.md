# Released Checkpoints

| File | Architecture | Source |
| --- | --- | --- |
| `flat_pretrain.pt` | Feed-forward PPO actor-critic with symmetry augmentation | `2026-07-28_21-31-50/model_2998.pt` |
| `rough_terrain_teacher.pt` | Privileged feed-forward PPO teacher | `2026-07-29_17-03-20/model_6997.pt` |
| `distilled_student.pt` | GRU teacher-student distillation checkpoint | `2026-07-29_19-44-17/model_2999.pt` |
| `distilled_policy.pt` | Exported TorchScript recurrent student | `2026-07-29_19-44-17/exported/policy.pt` |

The teacher receives 247 observations. The deployable student receives 57 observations and uses `GRU(57, 247)` followed by an MLP with hidden dimensions `[512, 256, 128]`.

Matching Hydra configurations are stored in `training_configs/flat`, `training_configs/teacher`, and `training_configs/student`.

Expected SHA256 hashes are listed in `SHA256SUMS.txt`.