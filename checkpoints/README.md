# Released Checkpoints

| File | Architecture | Source |
| --- | --- | --- |
| `flat_pretrain.pt` | Feed-forward PPO actor-critic with symmetry augmentation | `2026-07-28_21-31-50/model_2998.pt` |
| `rough_terrain_teacher.pt` | Privileged feed-forward PPO teacher | `2026-07-29_17-03-20/model_6997.pt` |
| `distilled_student.pt` | GRU teacher-student distillation checkpoint | `2026-07-29_19-44-17/model_2999.pt` |
| `distilled_policy.pt` | Exported TorchScript recurrent student | `2026-07-29_19-44-17/exported/policy.pt` |
| [`distilled_policy.onnx`](distilled_policy.onnx) | Exported ONNX recurrent student | `2026-07-29_19-44-17/exported/policy.onnx` |

The teacher receives 247 observations. The deployable student receives 57 observations and uses `GRU(57, 247)` followed by an MLP with hidden dimensions `[512, 256, 128]`.

Matching Hydra configurations are stored in `training_configs/flat`, `training_configs/teacher`, and `training_configs/student`.

Expected SHA256 hashes are listed in `SHA256SUMS.txt`.

## ONNX Inference

`distilled_policy.onnx` exports the same successful student as `distilled_policy.pt`. It is an inference artifact, not a checkpoint for resuming training. All tensors are `float32`; this export has a fixed batch size of 1.

| Direction | Name | Shape | Meaning |
| --- | --- | --- | --- |
| Input | `obs` | `[1, 57]` | Student observation in training order and scale |
| Input | `h_in` | `[1, 1, 247]` | Previous GRU hidden state |
| Output | `actions` | `[1, 12]` | Raw policy actions, not motor torques |
| Output | `h_out` | `[1, 1, 247]` | Updated GRU hidden state |

Install the CPU runtime with `python -m pip install onnxruntime numpy`. From the repository root, the following is a minimal interface smoke test, not a robot controller:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "checkpoints/distilled_policy.onnx", providers=["CPUExecutionProvider"]
)
hidden = np.zeros((1, 1, 247), dtype=np.float32)
obs = np.zeros((1, 57), dtype=np.float32)  # Replace with actual processed observations.
actions, hidden = session.run(
    ["actions", "h_out"], {"obs": obs, "h_in": hidden}
)
print(actions.shape, hidden.shape)
```

Feed `h_out` back as `h_in` on the next policy step. Reset the hidden state to zeros when resetting the robot/episode, not on every step. Match observation definitions, joint ordering, action scaling, and PD control to `sim2sim/run_policy_flat.py`.

The existing MuJoCo runner uses `torch.jit.load` and still requires `distilled_policy.pt`; passing this ONNX file to its `--policy` option is not supported.
