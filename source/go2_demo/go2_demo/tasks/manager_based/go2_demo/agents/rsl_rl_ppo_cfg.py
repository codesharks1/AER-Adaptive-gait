# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .. import mdp


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_demo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=mdp.compute_symmetric_states,
        ),
    )


@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the full-observation teacher into a recurrent deployable student."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "go2_demo"
    obs_groups = {
        "policy": ["student"],
        "teacher": ["policy"],
    }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        # The distillation implementation builds the non-recurrent teacher MLP with this width.
        # Keep it equal to the teacher actor input width so teacher checkpoints load correctly.
        rnn_hidden_dim=247,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="huber",
    )


@configclass
class StudentFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Fine-tune the distilled recurrent student with PPO rewards."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "go2_demo"
    obs_groups = {
        "policy": ["student"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=247,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )