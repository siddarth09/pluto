"""Domain randomisation, with ranges set by the identifiability analysis.

mjlab's velocity task randomises foot friction, encoder bias, base CoM, pushes
and reset pose, but nothing about the joints or actuators. These five terms are
additive.

Widths come from how well each parameter was actually determined -- see
sysid/results/g1_legs_identified.json. Scoped to leg joints: nothing above the
pelvis was identified.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

HIP_JOINTS = (r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint")
KNEE_ANKLE_JOINTS = (r".*_knee_joint", r".*_ankle_pitch_joint", r".*_ankle_roll_joint")
LEG_JOINTS = HIP_JOINTS + KNEE_ANKLE_JOINTS
MODE = "startup"


def add_identified_randomisation(
    cfg: ManagerBasedRlEnvCfg,
    gain: tuple[float, float] = (0.95, 1.05),
    armature_rel: tuple[float, float] = (0.5, 3.0),
    frictionloss_abs: tuple[float, float] = (0.2, 2.0),
    damping_hips_rel: tuple[float, float] = (0.7, 1.3),
    damping_other_abs: tuple[float, float] = (0.01, 2.0),
    mode: str = MODE,
) -> ManagerBasedRlEnvCfg:
    """Add actuator and joint randomisation to a velocity env cfg, in place."""

    cfg.events["actuator_gain"] = EventTermCfg(
        func=dr.pd_gains,
        mode=mode,
        params={
            # Scales the identified stiffness/damping, so this is uncertainty
            # ABOUT the measured gain, not a re-guess of it.
            "kp_range": gain,
            "kd_range": gain,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["joint_armature"] = EventTermCfg(
        func=dr.joint_armature,
        mode=mode,
        params={
            "ranges": armature_rel,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )
    cfg.events["joint_frictionloss"] = EventTermCfg(
        func=dr.joint_friction,
        mode=mode,
        params={
            "ranges": frictionloss_abs,
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )
    cfg.events["joint_damping_hips"] = EventTermCfg(
        func=dr.joint_damping,
        mode=mode,
        params={
            "ranges": damping_hips_rel,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINTS),
        },
    )
    cfg.events["joint_damping_knee_ankle"] = EventTermCfg(
        func=dr.joint_damping,
        mode=mode,
        params={
            "ranges": damping_other_abs,
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_ANKLE_JOINTS),
        },
    )
    return cfg
