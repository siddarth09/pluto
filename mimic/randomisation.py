"""Arm and waist randomisation, additive to the leg terms in mjlab_g1.

Widths follow the identifiability verdicts in sysid/results/. Tracking drives the
arms hard, so unlike the velocity task these cannot be left out.

  actuator gain   identified near unity; nominal gain stands, range is its error
  frictionloss    measured, except shoulder_roll, which gets its own wide range
  armature        not identified; motor spec stands, range is wide
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
    # Waist was never identified -- clamped at full gain during the arm run.
    # These widths are ignorance, not measurement.
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
