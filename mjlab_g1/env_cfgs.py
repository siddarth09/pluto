"""Velocity-task configs using the system-identified G1.

Thin wrapper over mjlab's own G1 velocity configs: build theirs, then swap in the
identified robot and the action scale that follows from its stiffness. Everything
else -- sensors, rewards, terrain, curriculum -- is inherited, so upstream fixes
arrive for free and only the actuator model differs.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)

from pluto.mjlab_g1.g1_identified import (
    get_g1_identified_robot_cfg,
    identified_action_scale,
)
from pluto.mjlab_g1.observations import make_actor_deployable
from pluto.mjlab_g1.randomisation import add_identified_randomisation
from pluto.mjlab_g1.rewards import apply_curriculum, apply_rewards

# Command latency randomised over 0..DELAY_MAX_LAG physics steps (5 ms each).
# 4 steps = 20 ms = one 50 Hz policy step.
DELAY_MAX_LAG = 4

# Hardware-measured command-to-response lag, 2026-09-23: 4 policy steps = 80 ms
# (cross-correlation minimum over the 12 leg joints, deploy_log.npz).
MEASURED_LAG_STEPS = 16  # 16 physics steps x 5 ms = 80 ms


def _swap_in_identified_robot(
    cfg: ManagerBasedRlEnvCfg, delay_max_lag: int = 0, delay_min_lag: int = 0
) -> ManagerBasedRlEnvCfg:
    # Safe to replace the entity after upstream has wired everything: sensors,
    # rewards and events all reference bodies/sites/geoms by name, and only the
    # actuator configs differ from upstream.
    cfg.scene.entities = {
        "robot": get_g1_identified_robot_cfg(delay_max_lag, delay_min_lag)
    }
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = identified_action_scale()
    apply_rewards(cfg)
    apply_curriculum(cfg)
    make_actor_deployable(cfg)
    return cfg


def pluto_g1_flat_env_cfg(
    play: bool = False, randomise: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = _swap_in_identified_robot(
        unitree_g1_flat_env_cfg(play=play), DELAY_MAX_LAG if randomise and not play else 0
    )
    # Randomisation is off in play mode: it would confound evaluation.
    if randomise and not play:
        add_identified_randomisation(cfg)
    return cfg


# Upstream sets nconmax=70 for the G1 rough task, which overflows ("nconmax must
# be >= 72") on the play-mode terrain. Contact buffers are cheap; give it slack.
ROUGH_NCONMAX = 200


def pluto_g1_flat_delay_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat, with the DETERMINISTIC measured hardware lag injected.

    For reproducing the hardware behaviour in sim, not for training: if a policy
    oscillates here the way it does on the robot, the lag is the cause.
    """
    return _swap_in_identified_robot(
        unitree_g1_flat_env_cfg(play=play),
        delay_max_lag=MEASURED_LAG_STEPS,
        delay_min_lag=MEASURED_LAG_STEPS,
    )


def pluto_g1_rough_env_cfg(
    play: bool = False, randomise: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = _swap_in_identified_robot(
        unitree_g1_rough_env_cfg(play=play), DELAY_MAX_LAG if randomise and not play else 0
    )
    cfg.sim.nconmax = ROUGH_NCONMAX
    if randomise and not play:
        add_identified_randomisation(cfg)
    return cfg
