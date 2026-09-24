"""PPO runner config, editable from PLUTO.

Wraps mjlab's G1 velocity runner config and overrides only the fields worth
tuning, so upstream defaults for everything else still apply.
"""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

# Network width/depth. Upstream is (512, 256, 128) for both actor and critic.
# The critic can afford to be larger than the actor -- it never runs on hardware.
ACTOR_HIDDEN_DIMS = (1024, 512, 256, 128)
CRITIC_HIDDEN_DIMS = (1024, 512, 256, 128)

# Observation history on the actor. With base_lin_vel dropped, the policy has to
# infer its own forward speed; a single timestep of joint state plus gravity may
# not be enough, and history is fully deployable. None = single timestep.
ACTOR_HISTORY_LENGTH: int | None = None

MAX_ITERATIONS = 120_000
SAVE_INTERVAL = 2000 # upstream 50; 120k iterations would otherwise write 2400 checkpoints


def pluto_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.actor = dataclasses.replace(cfg.actor, hidden_dims=ACTOR_HIDDEN_DIMS)
    cfg.critic = dataclasses.replace(cfg.critic, hidden_dims=CRITIC_HIDDEN_DIMS)
    cfg.max_iterations = MAX_ITERATIONS
    cfg.save_interval = SAVE_INTERVAL
    cfg.experiment_name = "pluto_g1_velocity_identified"
    return cfg
