"""Domain randomisation sized by the system identification.

mjlab's velocity task randomises foot friction, encoder bias, base CoM, pushes and
reset pose -- but nothing about the joints or actuators. That is the gap the
identification fills, and it matters most: a systematic actuator-gain error is in
the same direction on every sample, so no amount of randomising other parameters
covers it.

Each range below is set by how well the parameter was actually determined, not by
a uniform guess. See pluto/sysid/README.md.

  actuator gain   IDENTIFIED (30x cost change over +-20%, two independent methods
                  within 6%, validated across a 2x gain change) -> tight, +-5%
  damping, hips   IDENTIFIED (left/right mirrored joints agree within 14%) -> +-30%
  damping, other  NOT identified (pinned at bound) -> wide, absolute
  frictionloss    NOT identifiable from motion; five fits spanned 0.001-1.16 with
                  <3% accuracy change -> wide, absolute, centred on the measured
                  static breakaway values
  armature        NOT identified (mirrored joints disagree 6-60x) -> wide, relative
                  to the motor spec

Only leg joints are scoped: nothing above the pelvis was identified, and applying
leg-derived ranges to the arms would be inventing data.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

HIP_JOINTS = (r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint")
KNEE_ANKLE_JOINTS = (r".*_knee_joint", r".*_ankle_pitch_joint", r".*_ankle_roll_joint")
LEG_JOINTS = HIP_JOINTS + KNEE_ANKLE_JOINTS

# mode="startup" samples once per environment, matching how mjlab's own
# foot_friction / encoder_bias / base_com terms work. Switching to "reset"
# resamples every episode, which covers the ranges far better for the same env
# count; the dr functions take env_ids so it is supported. Left at startup to
# match the pattern already known to work in this codebase.
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
