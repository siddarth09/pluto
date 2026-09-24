"""G1 locomotion on the system-identified actuator model.

Importing this module registers the tasks. Use the entrypoints in this package
(`train.py`, `play.py`) so the import happens before the task lookup.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from pluto.mjlab_g1.env_cfgs import (
    pluto_g1_flat_delay_env_cfg,
    pluto_g1_flat_env_cfg,
    pluto_g1_rough_env_cfg,
)
from pluto.mjlab_g1.g1_identified import (
    RANDOMISATION,
    get_g1_identified_robot_cfg,
    identified_action_scale,
)

from pluto.mjlab_g1.rl_cfg import pluto_g1_ppo_runner_cfg

FLAT_TASK = "Pluto-Velocity-Flat-G1-Identified"
FLAT_DR_TASK = "Pluto-Velocity-Flat-G1-Identified-DR"
ROUGH_TASK = "Pluto-Velocity-Rough-G1-Identified"
ROUGH_DR_TASK = "Pluto-Velocity-Rough-G1-Identified-DR"
FLAT_DELAY_TASK = "Pluto-Velocity-Flat-G1-Identified-Delay80"

# Registered with and without randomisation so the two can be compared. The
# non-DR variants are the debugging baseline; the DR variants are what you would
# deploy from.
for _task_id, _make, _randomise in (
    (FLAT_TASK, pluto_g1_flat_env_cfg, False),
    (FLAT_DR_TASK, pluto_g1_flat_env_cfg, True),
    (ROUGH_TASK, pluto_g1_rough_env_cfg, False),
    (ROUGH_DR_TASK, pluto_g1_rough_env_cfg, True),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_make(randomise=_randomise),
        play_env_cfg=_make(play=True),
        rl_cfg=pluto_g1_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )

register_mjlab_task(
    task_id=FLAT_DELAY_TASK,
    env_cfg=pluto_g1_flat_delay_env_cfg(),
    play_env_cfg=pluto_g1_flat_delay_env_cfg(play=True),
    rl_cfg=pluto_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

__all__ = [
    "FLAT_TASK",
    "FLAT_DR_TASK",
    "FLAT_DELAY_TASK",
    "ROUGH_TASK",
    "ROUGH_DR_TASK",
    "RANDOMISATION",
    "get_g1_identified_robot_cfg",
    "identified_action_scale",
    "pluto_g1_flat_env_cfg",
    "pluto_g1_rough_env_cfg",
]
