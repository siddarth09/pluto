"""Motion tracking on the identified G1. Override of mjlab's tracking task.

Changes from `unitree_g1_flat_tracking_env_cfg`: identified robot cfg and action
scale, the motion file, and joint/actuator randomisation.

has_state_estimation defaults False: no base velocity estimate on hardware, so
`base_lin_vel` and `motion_anchor_pos_b` are dropped from the actor.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from pluto.mimic.randomisation import add_arm_randomisation
from pluto.mjlab_g1.g1_identified import (
    get_g1_identified_robot_cfg,
    identified_action_scale,
)
from pluto.mjlab_g1.randomisation import add_identified_randomisation

MOTION_DIR = Path(__file__).resolve().parent / "motions"
DEFAULT_MOTION = MOTION_DIR / "J_Dance17_Shuffle_50hz.npz"

# Measured command->response lag is 60-80 ms. 4 steps at 50 Hz = 80 ms.
DELAY_MAX_LAG = 4
DELAY_MIN_LAG = 0


def _delay_every_actuator(robot_cfg, min_lag: int, max_lag: int):
    """Delay every actuator group, not just the legs.

    get_g1_identified_robot_cfg only delays _leg_actuators. Tracking drives all
    29 joints and the lag applies to all of them.
    """
    art = robot_cfg.articulation
    acts = tuple(
        dataclasses.replace(a, delay_min_lag=min_lag, delay_max_lag=max_lag)
        for a in art.actuators
    )
    return dataclasses.replace(
        robot_cfg, articulation=dataclasses.replace(art, actuators=acts)
    )


def pluto_g1_mimic_env_cfg(
    motion_file: Path | str = DEFAULT_MOTION,
    randomise: bool = True,
    has_state_estimation: bool = False,
    delay: bool = True,
    track_global_root_pos: bool = False,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Tracking cfg for one clip on the identified G1."""
    motion_file = Path(motion_file)
    if not motion_file.is_file():
        raise SystemExit(
            f"motion not found: {motion_file}\n"
            "run ./prep_motion.sh: MotionLoader does not resample, so a 60 fps "
            "clip plays 1.2x slow at 50 Hz."
        )

    cfg = unitree_g1_flat_tracking_env_cfg(
        has_state_estimation=has_state_estimation, play=play
    )

    robot = get_g1_identified_robot_cfg()
    if delay:
        robot = _delay_every_actuator(robot, DELAY_MIN_LAG, DELAY_MAX_LAG)
    cfg.scene.entities = {"robot": robot}
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = identified_action_scale()

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_file)

    # Global root position is unobservable without base pos or base lin vel, so
    # the term only adds variance. Orientation stays: motion_anchor_ori_b sees it.
    if not track_global_root_pos and not has_state_estimation:
        cfg.rewards["motion_global_root_pos"].weight = 0.0

    if randomise and not play:
        add_identified_randomisation(cfg)
        add_arm_randomisation(cfg)

    return cfg
