from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse


class FootTouchdownLogger:
    """Log foot touchdown positions relative to the robot base."""

    FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    def __init__(self, env, env_index: int = 0):
        if env_index < 0 or env_index >= env.num_envs:
            raise ValueError(f"touchdown_env must be in [0, {env.num_envs - 1}], received {env_index}.")

        self._env = env
        self._env_index = env_index
        self._robot = env.scene["robot"]
        self._contact_sensor = env.scene.sensors["contact_forces"]

        requested_names = list(self.FOOT_NAMES)
        self._body_ids, body_names = self._robot.find_bodies(requested_names, preserve_order=True)
        self._sensor_body_ids, sensor_names = self._contact_sensor.find_bodies(
            requested_names, preserve_order=True
        )
        if body_names != requested_names or sensor_names != requested_names:
            raise RuntimeError(
                "Foot order could not be resolved as FL, FR, RL, RR. "
                f"Robot={body_names}, contact_sensor={sensor_names}."
            )

        self._position_sums = torch.zeros((len(self.FOOT_NAMES), 3), dtype=torch.float64)
        self._touchdown_counts = torch.zeros(len(self.FOOT_NAMES), dtype=torch.long)
        self._steps_since_reset = 0

        print(
            f"[TOUCHDOWN] Logging env {env_index}; coordinates are relative to the base "
            "(x=forward, y=left, z=up)."
        )

    def update(self, step: int, dones: torch.Tensor) -> None:
        """Record first-contact positions for the selected environment."""
        if bool(dones[self._env_index].item()):
            self._steps_since_reset = 0
            return

        self._steps_since_reset += 1
        if self._steps_since_reset <= 2:
            return

        first_contact = self._contact_sensor.compute_first_contact(self._env.step_dt)[
            self._env_index, self._sensor_body_ids
        ]
        if not bool(first_contact.any().item()):
            return

        foot_pos_w = self._robot.data.body_pos_w[self._env_index, self._body_ids]
        base_pos_w = self._robot.data.root_pos_w[self._env_index]
        foot_pos_rel_w = foot_pos_w - base_pos_w.unsqueeze(0)
        base_quat_w = self._robot.data.root_quat_w[self._env_index].unsqueeze(0).expand(len(self.FOOT_NAMES), -1)
        foot_pos_b = quat_apply_inverse(base_quat_w, foot_pos_rel_w)

        for foot_index in torch.nonzero(first_contact, as_tuple=False).flatten().tolist():
            position = foot_pos_b[foot_index].detach().cpu()
            self._position_sums[foot_index] += position.to(dtype=torch.float64)
            self._touchdown_counts[foot_index] += 1
            print(
                f"[TOUCHDOWN] step={step:06d} env={self._env_index} foot={self.FOOT_NAMES[foot_index]} "
                f"x={position[0].item(): .4f} y={position[1].item(): .4f} z={position[2].item(): .4f}"
            )

    def print_summary(self) -> None:
        """Print mean touchdown positions collected during play."""
        print(f"[TOUCHDOWN SUMMARY] env={self._env_index}")
        means: dict[str, torch.Tensor] = {}
        for foot_index, foot_name in enumerate(self.FOOT_NAMES):
            count = int(self._touchdown_counts[foot_index].item())
            if count == 0:
                print(f"  {foot_name}: no touchdown samples")
                continue

            mean_position = self._position_sums[foot_index] / count
            means[foot_name] = mean_position
            print(
                f"  {foot_name}: n={count:4d} "
                f"mean_x={mean_position[0].item(): .4f} "
                f"mean_y={mean_position[1].item(): .4f} "
                f"mean_z={mean_position[2].item(): .4f}"
            )

        if "RL_foot" in means and "RR_foot" in means:
            rear_delta_x = means["RL_foot"][0] - means["RR_foot"][0]
            print(f"  rear_delta_x (RL - RR)={rear_delta_x.item(): .4f} m")
