"""PPO config for the tracking task.

Wider than mjlab's (512,256,128) default: the actor maps a reference pose plus
its own state to 29 joint targets.
"""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

ACTOR_HIDDEN_DIMS = (1024, 512, 256, 128)
CRITIC_HIDDEN_DIMS = (1024, 512, 256, 128)
MAX_ITERATIONS = 30_000
SAVE_INTERVAL = 1_000


def pluto_g1_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=ACTOR_HIDDEN_DIMS,
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=CRITIC_HIDDEN_DIMS,
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="pluto_g1_mimic_dance17",
        save_interval=SAVE_INTERVAL,
        num_steps_per_env=24,
        max_iterations=MAX_ITERATIONS,
    )
