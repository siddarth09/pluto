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
from pluto.mjlab_g1.randomisation import add_identified_randomisation


def _swap_in_identified_robot(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    # Safe to replace the entity after upstream has wired everything: sensors,
    # rewards and events all reference bodies/sites/geoms by name, and only the
    # actuator configs differ from upstream.
    cfg.scene.entities = {"robot": get_g1_identified_robot_cfg()}
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = identified_action_scale()
    return cfg


def pluto_g1_flat_env_cfg(
    play: bool = False, randomise: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = _swap_in_identified_robot(unitree_g1_flat_env_cfg(play=play))
    # Randomisation is off in play mode: it would confound evaluation.
    if randomise and not play:
        add_identified_randomisation(cfg)
    return cfg


def pluto_g1_rough_env_cfg(
    play: bool = False, randomise: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = _swap_in_identified_robot(unitree_g1_rough_env_cfg(play=play))
    if randomise and not play:
        add_identified_randomisation(cfg)
    return cfg
