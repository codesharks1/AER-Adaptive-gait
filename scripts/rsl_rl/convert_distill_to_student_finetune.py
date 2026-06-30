# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a distilled recurrent student checkpoint into a PPO recurrent actor checkpoint.

The distillation checkpoint stores deployable policy weights under::

    memory_s.*
    student.*

The PPO recurrent fine-tuning task expects the same policy under::

    memory_a.*
    actor.*

This script creates a full ActorCriticRecurrent checkpoint. The actor is initialized from
``memory_s`` and ``student``. The critic is freshly initialized and will be learned during PPO fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticRecurrent


def _copy_prefixed(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor], source_prefix: str, target_prefix: str) -> None:
    for source_key, value in source_state.items():
        if not source_key.startswith(source_prefix):
            continue
        target_key = target_prefix + source_key[len(source_prefix) :]
        if target_key not in target_state:
            raise KeyError(f"Converted key is not present in target model: {target_key}")
        if target_state[target_key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {target_key}: target {tuple(target_state[target_key].shape)} vs "
                f"source {tuple(value.shape)}"
            )
        target_state[target_key] = value.clone()


def convert(args: argparse.Namespace) -> None:
    distill_checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    distill_state = distill_checkpoint["model_state_dict"]

    fake_obs = TensorDict(
        {
            "student": torch.zeros(1, args.student_obs_dim),
            "critic": torch.zeros(1, args.critic_obs_dim),
        },
        batch_size=[1],
    )
    policy = ActorCriticRecurrent(
        obs=fake_obs,
        obs_groups={"policy": ["student"], "critic": ["critic"]},
        num_actions=args.num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=args.hidden_dims,
        critic_hidden_dims=args.hidden_dims,
        activation="elu",
        init_noise_std=args.init_noise_std,
        rnn_type="gru",
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_num_layers=1,
    )

    target_state = policy.state_dict()
    _copy_prefixed(distill_state, target_state, "memory_s.", "memory_a.")
    _copy_prefixed(distill_state, target_state, "student.", "actor.")

    if "std" in distill_state and "std" in target_state:
        target_state["std"] = distill_state["std"].clone()
    if "log_std" in distill_state and "log_std" in target_state:
        target_state["log_std"] = distill_state["log_std"].clone()

    policy.load_state_dict(target_state, strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "source_distillation_checkpoint": str(Path(args.input).resolve()),
                "note": "Actor initialized from distilled student; critic freshly initialized for PPO fine-tuning.",
            },
        },
        output,
    )
    print(f"[INFO] Wrote converted student fine-tuning checkpoint to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the distillation checkpoint, e.g. model_2999.pt")
    parser.add_argument("--output", required=True, help="Path for the converted PPO recurrent checkpoint")
    parser.add_argument("--student_obs_dim", type=int, default=57)
    parser.add_argument("--critic_obs_dim", type=int, default=247)
    parser.add_argument("--num_actions", type=int, default=12)
    parser.add_argument("--rnn_hidden_dim", type=int, default=247)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--init_noise_std", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()