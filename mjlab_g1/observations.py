"""Observation-space edits for hardware deployability.

The actor observation is the only thing that has to exist on the robot. The critic
keeps privileged simulator state (true base velocity, foot contact, air time,
contact forces) -- that is asymmetric actor-critic and is discarded at deployment.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

# `unitree_hg.LowState_` carries only quaternion, gyroscope, accelerometer, rpy and
# motor states -- there is no base linear velocity. The robot does estimate it
# internally to balance in `ai` mode, but it is not published in the low-level
# message set, and ReleaseMode() drops the service that computes it. So the actor
# cannot have it. Everything else it observes is a direct IMU or encoder reading.
DROP_FROM_ACTOR = ("base_lin_vel",)

# Stacked past observations on the actor. Gives the network the temporal signal it
# needs to infer forward speed now that base_lin_vel is gone, and is entirely
# deployable. None = single timestep (upstream behaviour).
ACTOR_HISTORY_LENGTH: int | None = 5


def make_actor_deployable(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    for term in DROP_FROM_ACTOR:
        cfg.observations["actor"].terms.pop(term, None)
    if ACTOR_HISTORY_LENGTH is not None:
        cfg.observations["actor"].history_length = ACTOR_HISTORY_LENGTH
    return cfg
