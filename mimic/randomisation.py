"""Arm and waist randomisation, on top of the leg terms from the velocity task.

Widths come from the arm identification of 2026-09-28 (sysid/results/
arm_stiction.npz and the tau_est gain regression), the same way the leg widths
came from the leg analysis. Tracking a whole-body motion uses the arms hard, so
unlike the velocity task these cannot be left out.

Verdicts that set these numbers:
- actuator gain: MEASURED at 0.92-1.06 across 12 of 14 joints, two independent
  recordings agreeing to 0.002. Unlike the legs there is no bias to correct, so
  the model keeps nominal gain and this is uncertainty about that.
- frictionloss: MEASURED 0.20-0.59 for six of seven mirrored pairs. shoulder_roll
  is NOT resolved (2.19 left, over the 3.75 ceiling right) and gets its own wide
  range.
- armature: NOT identified. Six values pinned at their lower bound and mirrored
  joints disagreed 2-150x, so the motor spec stands and the range is wide.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

ARM_JOINTS = (
    r".*_shoulder_pitch_joint",
    r".*_shoulder_yaw_joint",
    r".*_elbow_joint",
    r".*_wrist_roll_joint",
    r".*_wrist_pitch_joint",
    r".*_wrist_yaw_joint",
)
SHOULDER_ROLL = (r".*_shoulder_roll_joint",)
WAIST_JOINTS = (r"waist_yaw_joint", r"waist_roll_joint", r"waist_pitch_joint")
MODE = "startup"


def add_arm_randomisation(
    cfg: ManagerBasedRlEnvCfg,
    arm_gain: tuple[float, float] = (0.92, 1.06),
    arm_armature_rel: tuple[float, float] = (0.5, 3.0),
    arm_friction_abs: tuple[float, float] = (0.15, 0.80),
    shoulder_roll_friction_abs: tuple[float, float] = (0.40, 2.50),
    waist_gain: tuple[float, float] = (0.75, 1.25),
    waist_armature_rel: tuple[float, float] = (0.5, 3.0),
    waist_friction_abs: tuple[float, float] = (0.10, 1.50),
    mode: str = MODE,
) -> ManagerBasedRlEnvCfg:
    """Add arm and waist randomisation to a tracking env cfg, in place."""

    cfg.events["arm_actuator_gain"] = EventTermCfg(
        func=dr.pd_gains,
        mode=mode,
        params={
            "kp_range": arm_gain,
            "kd_range": arm_gain,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS + SHOULDER_ROLL),
        },
    )
    cfg.events["arm_armature"] = EventTermCfg(
        func=dr.joint_armature,
        mode=mode,
        params={
            "ranges": arm_armature_rel,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS + SHOULDER_ROLL),
        },
    )
    cfg.events["arm_frictionloss"] = EventTermCfg(
        func=dr.joint_friction,
        mode=mode,
        params={
            "ranges": arm_friction_abs,
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )
    cfg.events["shoulder_roll_frictionloss"] = EventTermCfg(
        func=dr.joint_friction,
        mode=mode,
        params={
            "ranges": shoulder_roll_friction_abs,
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=SHOULDER_ROLL),
        },
    )
    # The waist was never identified: it was clamped at full gain to make the
    # arm experiment valid, so every width here is ignorance, not measurement.
    cfg.events["waist_actuator_gain"] = EventTermCfg(
        func=dr.pd_gains,
        mode=mode,
        params={
            "kp_range": waist_gain,
            "kd_range": waist_gain,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS),
        },
    )
    cfg.events["waist_armature"] = EventTermCfg(
        func=dr.joint_armature,
        mode=mode,
        params={
            "ranges": waist_armature_rel,
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS),
        },
    )
    cfg.events["waist_frictionloss"] = EventTermCfg(
        func=dr.joint_friction,
        mode=mode,
        params={
            "ranges": waist_friction_abs,
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS),
        },
    )
    return cfg
