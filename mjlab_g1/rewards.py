"""Reward weights, pose tolerances and curriculum, editable from PLUTO.

Upstream builds the reward terms in mjlab's `make_velocity_env_cfg` and tunes them
per robot in `tasks/velocity/config/g1/env_cfgs.py`. Redefining the terms here
would duplicate their params and drift from upstream; instead the inherited terms
are kept and only weights, tolerances and curriculum stages are overridden.

To add a NEW reward term, put it in `extra_reward_terms()`.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: F401  for new terms

# Inherited reward weights. Values are upstream's G1 defaults; change them here.
#   + rewards the behaviour, - penalises it.
WEIGHTS: dict[str, float] = {
    # Task: what the policy is actually asked to do.
    "track_linear_velocity": 2.0,
    "track_angular_velocity": 2.0,
    # Posture and stability.
    "upright": 1.0,
    "pose": 1.0,          # stay near the default pose, per-joint std below
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "self_collisions": -1.0,
    # Actuation and limits.
    "dof_pos_limits": -1.0,
    "action_rate_l2": -0.01,
    # Gait shaping. All four use contact sensors, which is fine: rewards and the
    # critic are training-only and never run on hardware.
    "air_time": 0.01,          # disabled upstream; raise to encourage longer strides
    "foot_clearance": -2.0,
    "foot_swing_height": -0.25,
    "foot_slip": -0.1,
    "soft_landing": -1e-03,
}

# Per-joint tolerance on the `pose` reward: larger std = more freedom to deviate
# from the default pose before being penalised. Regex keys.
POSE_STD_WALKING: dict[str, float] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
}

VELOCITY_STAGES = [
      {"step": 0,         "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
      {"step": 480_000,   "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},   # iter 20k
      {"step": 960_000,   "lin_vel_x": (-2.0, 3.0)},                             # iter 40k
  ]



def extra_reward_terms() -> dict[str, RewardTermCfg]:
    """New reward terms to add on top of the inherited ones."""
    return {}


def apply_rewards(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    for name, weight in WEIGHTS.items():
        if name not in cfg.rewards:
            raise KeyError(
                f"reward '{name}' is not in the inherited set {sorted(cfg.rewards)}; "
                "add it via extra_reward_terms() instead"
            )
        cfg.rewards[name].weight = weight
    cfg.rewards["pose"].params["std_walking"] = POSE_STD_WALKING
    cfg.rewards.update(extra_reward_terms())
    return cfg


def apply_curriculum(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    if "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = VELOCITY_STAGES
    return cfg
