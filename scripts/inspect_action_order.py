"""Inspect the resolved IsaacLab action-to-joint mapping for a task."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Print the model action index assigned to every controlled joint.")
parser.add_argument("--task", type=str, default="Go2-velocity-StudentFinetune-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra should only receive arguments that it understands.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_demo.tasks  # noqa: F401


def _as_first_env_list(value: torch.Tensor | float) -> list[float]:
    """Convert an action-term scalar or tensor into one value per action dimension."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 1:
            value = value[0]
        return value.detach().cpu().flatten().tolist()
    return [float(value)]


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg=None) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    print("\n" + "=" * 88)
    print(f"Task: {args_cli.task}")
    print("Robot articulation joint order:")
    for index, joint_name in enumerate(base_env.scene["robot"].joint_names):
        print(f"  robot.joint_names[{index:02d}] = {joint_name}")

    print("\nResolved action terms:")
    action_offset = 0
    for term_name in base_env.action_manager.active_terms:
        term = base_env.action_manager.get_term(term_name)
        descriptor = term.IO_descriptor
        joint_names = list(descriptor.joint_names)
        scales = _as_first_env_list(term._scale)
        offsets = _as_first_env_list(term._offset)
        if len(scales) == 1:
            scales *= len(joint_names)
        if len(offsets) == 1:
            offsets *= len(joint_names)

        print(f"\n  Term: {term_name} (dimension={term.action_dim})")
        print("  model_dim  joint_name             scale      default_offset   target formula")
        print("  ---------  ---------------------  ---------  ---------------  -------------------------------")
        for local_index, (joint_name, scale, offset) in enumerate(zip(joint_names, scales, offsets)):
            model_index = action_offset + local_index
            print(
                f"  action[{model_index:02d}]  {joint_name:21s}  {scale:9.4f}  {offset:15.4f}  "
                f"q_target = {offset:.4f} + {scale:.4f} * action[{model_index:02d}]"
            )
        action_offset += term.action_dim

    print("\nObservation/action tensors that use the articulation order should be checked against the list above.")
    print("=" * 88 + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
