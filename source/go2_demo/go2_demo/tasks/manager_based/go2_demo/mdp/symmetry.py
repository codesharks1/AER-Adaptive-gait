"""Left-right symmetry transforms for Go2 PPO training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


# Isaac Lab resolves the Go2 joints in this order:
# FL/FR/RL/RR hip, FL/FR/RL/RR thigh, FL/FR/RL/RR calf.
_LEFT_RIGHT_JOINT_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append a left-right mirrored copy of each observation and action.

    This encodes equivariance rather than a fixed gait: a mirrored robot state
    should produce the mirrored action, while the policy remains free to choose
    its timing and gait.
    """
    base_env = env.unwrapped

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        for group_name in obs.keys():
            group_obs = obs[group_name]
            obs_aug[group_name][:batch_size] = group_obs

            if group_name in base_env.observation_manager.active_terms:
                obs_aug[group_name][batch_size:] = _mirror_observation_group(
                    base_env, group_name, group_obs
                )
            else:
                obs_aug[group_name][batch_size:] = group_obs
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            (batch_size * 2, actions.shape[1]),
            dtype=actions.dtype,
            device=actions.device,
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _mirror_joint_data(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_observation_group(
    env: ManagerBasedRLEnv,
    group_name: str,
    obs: torch.Tensor,
) -> torch.Tensor:
    """Mirror one concatenated observation group using its configured terms."""
    mirrored = obs.clone()
    term_names = env.observation_manager.active_terms[group_name]
    term_dims = env.observation_manager.group_obs_term_dim[group_name]
    cursor = 0

    for term_name, term_shape in zip(term_names, term_dims):
        term_width = math.prod(term_shape)
        term_slice = slice(cursor, cursor + term_width)
        term = obs[..., term_slice]

        if term_name == "base_lin_vel":
            mirrored[..., term_slice] = term * term.new_tensor([1.0, -1.0, 1.0])
        elif term_name == "base_ang_vel":
            mirrored[..., term_slice] = term * term.new_tensor([-1.0, 1.0, -1.0])
        elif term_name == "projected_gravity":
            mirrored[..., term_slice] = term * term.new_tensor([1.0, -1.0, 1.0])
        elif term_name == "velocity_commands":
            mirrored[..., term_slice] = term * term.new_tensor([1.0, -1.0, -1.0])
        elif term_name in {"joint_pos_rel", "joint_vel_rel", "joint_effort", "last_action"}:
            mirrored[..., term_slice] = _mirror_joint_data(term)
        elif term_name == "height_scanner":
            mirrored[..., term_slice] = _mirror_height_scan(term)
        else:
            raise ValueError(
                f"No left-right symmetry rule is defined for observation term "
                f"'{group_name}.{term_name}'."
            )

        cursor += term_width

    if cursor != obs.shape[-1]:
        raise ValueError(
            f"Observation width mismatch for group '{group_name}': "
            f"transformed {cursor}, received {obs.shape[-1]}."
        )

    return mirrored


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and reverse hip-abduction signs."""
    if joint_data.shape[-1] != 12:
        raise ValueError(f"Expected 12 Go2 joint values, received {joint_data.shape[-1]}.")

    permutation = torch.tensor(
        _LEFT_RIGHT_JOINT_PERMUTATION,
        dtype=torch.long,
        device=joint_data.device,
    )
    mirrored = joint_data.index_select(-1, permutation).clone()
    mirrored[..., :4] *= -1.0
    return mirrored


def _mirror_height_scan(height_scan: torch.Tensor) -> torch.Tensor:
    """Flip the 11-by-17 base height grid across the robot's left-right axis."""
    if height_scan.shape[-1] != 187:
        raise ValueError(f"Expected a 187-value height scan, received {height_scan.shape[-1]}.")

    grid = height_scan.reshape(*height_scan.shape[:-1], 11, 17)
    return grid.flip(dims=[-2]).reshape_as(height_scan)
