"""Whole-body motion tracking on the system-identified G1.

Importing this module registers the tasks. Use the entrypoints in this package
(`train.py`, `play.py`) so the import happens before the task lookup.

Registered:
  Pluto-Mimic-Dance17-G1                 no randomisation, no delay: the baseline
                                         you debug against
  Pluto-Mimic-Dance17-G1-DR              randomisation + 80 ms delay: train this
  Pluto-Mimic-Dance17-G1-StateEst        with base_lin_vel in the actor; not
                                         deployable, useful as an upper bound
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from pluto.mimic.env_cfgs import DEFAULT_MOTION, pluto_g1_mimic_env_cfg
from pluto.mimic.rl_cfg import pluto_g1_mimic_ppo_runner_cfg

BASE_TASK = "Pluto-Mimic-Dance17-G1"
DR_TASK = "Pluto-Mimic-Dance17-G1-DR"
STATE_EST_TASK = "Pluto-Mimic-Dance17-G1-StateEst"

for _task_id, _kwargs in (
    (BASE_TASK, dict(randomise=False, delay=False)),
    (DR_TASK, dict(randomise=True, delay=True)),
    (STATE_EST_TASK, dict(randomise=True, delay=True, has_state_estimation=True)),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=pluto_g1_mimic_env_cfg(**_kwargs),
        play_env_cfg=pluto_g1_mimic_env_cfg(
            **{**_kwargs, "randomise": False}, play=True
        ),
        rl_cfg=pluto_g1_mimic_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

__all__ = [
    "BASE_TASK",
    "DR_TASK",
    "STATE_EST_TASK",
    "DEFAULT_MOTION",
    "pluto_g1_mimic_env_cfg",
    "pluto_g1_mimic_ppo_runner_cfg",
]
