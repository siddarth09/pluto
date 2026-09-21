"""G1 entity config with system-identified leg parameters.

Overrides mjlab's asset-zoo G1 without modifying it: import the upstream config,
replace the leg actuator groups with per-joint configs carrying the identified
values, and leave the upper body untouched.

Source of the numbers: pluto/sysid/ (see its README). Held-out open-loop replay
error 0.0251 rad vs 0.0365 rad nominal, and 0.0246 rad on a second holdout
recorded at double the servo gains.

Per-joint configs are necessary because BuiltinPositionActuatorCfg takes scalar
stiffness and damping, while the measured actuator gain scale varies within
mjlab's motor groups (hip_pitch 0.952 vs hip_yaw 0.860, both 7520_14).

    from pluto.mjlab_g1.g1_identified import get_g1_identified_robot_cfg
    robot = get_g1_identified_robot_cfg()
"""

from __future__ import annotations

import dataclasses

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_WAIST,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    get_g1_robot_cfg,
)
from mjlab.entity import EntityCfg

# Measured on hardware, hoisted, 2026-09-21. Order: hip_pitch, hip_roll, hip_yaw,
# knee, ankle_pitch, ankle_roll -- identical for both legs except where noted.
#
# gain_scale     measured tau_est / commanded PD torque. IDENTIFIED: 30x cost
#                change over +-20%, validated across a 2x gain change.
# frictionloss   static breakaway from gravity holds. NOT fittable from motion
#                (five fits spanned 0.001-1.16 with <3% accuracy change).
# viscous_damping fitted. Identified on the hips (left/right agree within 14%),
#                NOT identified on knee/ankle_roll (pinned at the bound).
# armature       motor spec. The fit is unusable here: mirrored joints disagree
#                6-60x, so the spec is strictly better information.
_LEG = {
    "hip_pitch": dict(gain=(0.952, 0.943), friction=(0.53, 0.53),
                      damp=(3.12337, 3.03246),
                      stiff=STIFFNESS_7520_14, kd=DAMPING_7520_14,
                      arm=ACTUATOR_7520_14.reflected_inertia,
                      effort=ACTUATOR_7520_14.effort_limit),
    "hip_roll": dict(gain=(0.936, 0.925), friction=(0.59, 0.59),
                     damp=(0.57747, 0.65896),
                     stiff=STIFFNESS_7520_22, kd=DAMPING_7520_22,
                     arm=ACTUATOR_7520_22.reflected_inertia,
                     effort=ACTUATOR_7520_22.effort_limit),
    "hip_yaw": dict(gain=(0.860, 0.847), friction=(0.16, 0.16),
                    damp=(0.35904, 0.32976),
                    stiff=STIFFNESS_7520_14, kd=DAMPING_7520_14,
                    arm=ACTUATOR_7520_14.reflected_inertia,
                    effort=ACTUATOR_7520_14.effort_limit),
    "knee": dict(gain=(0.820, 0.857), friction=(0.67, 0.85),
                 damp=(0.001, 0.001),
                 stiff=STIFFNESS_7520_22, kd=DAMPING_7520_22,
                 arm=ACTUATOR_7520_22.reflected_inertia,
                 effort=ACTUATOR_7520_22.effort_limit),
    # Ankles are a 4-bar parallel linkage driven by two 5020s, so mjlab doubles
    # their nominal armature, stiffness and effort. Their gain scale is left at
    # 1.0: per-motor tau_est does not map onto pitch/roll joint torque through
    # the linkage, so the regression there was not credible (r 0.21-0.90).
    "ankle_pitch": dict(gain=(0.755, 1.0), friction=(0.21, 0.20),
                        damp=(1.4569, 0.80775),
                        stiff=STIFFNESS_5020 * 2, kd=DAMPING_5020 * 2,
                        arm=ACTUATOR_5020.reflected_inertia * 2,
                        effort=ACTUATOR_5020.effort_limit * 2),
    "ankle_roll": dict(gain=(1.0, 1.0), friction=(0.21, 0.21),
                       damp=(0.3929, 0.001),
                       stiff=STIFFNESS_5020 * 2, kd=DAMPING_5020 * 2,
                       arm=ACTUATOR_5020.reflected_inertia * 2,
                       effort=ACTUATOR_5020.effort_limit * 2),
}


def _leg_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    out = []
    for joint, p in _LEG.items():
        for side_index, side in enumerate(("left", "right")):
            g = p["gain"][side_index]
            out.append(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(f"{side}_{joint}_joint",),
                    # The robot delivers only `g` of the commanded torque, so the
                    # simulated servo is scaled down to match. Without this the
                    # policy learns an action-to-torque mapping the hardware does
                    # not have -- a bias no other parameter can absorb.
                    stiffness=p["stiff"] * g,
                    damping=p["kd"] * g,
                    effort_limit=p["effort"],
                    armature=p["arm"],
                    frictionloss=p["friction"][side_index],
                    viscous_damping=p["damp"][side_index],
                )
            )
    return tuple(out)


def _waist_yaw_only() -> BuiltinPositionActuatorCfg:
    """Upstream groups waist_yaw with the hip joints; the hips are now per-joint."""
    return dataclasses.replace(
        G1_ACTUATOR_7520_14, target_names_expr=("waist_yaw_joint",)
    )


def get_g1_identified_robot_cfg() -> EntityCfg:
    cfg = get_g1_robot_cfg()
    actuators = (
        G1_ACTUATOR_5020,     # arms, unchanged
        G1_ACTUATOR_4010,     # wrists, unchanged
        G1_ACTUATOR_WAIST,    # waist pitch/roll, unchanged
        _waist_yaw_only(),
        *_leg_actuators(),
    )
    cfg.articulation = dataclasses.replace(cfg.articulation, actuators=actuators)
    return cfg


def identified_action_scale() -> dict[str, float]:
    """Action scale recomputed from the identified stiffness.

    Upstream derives this as 0.25 * effort_limit / stiffness -- the position offset
    that commands a quarter of peak torque. Since the gain correction lowers
    stiffness, reusing upstream's G1_ACTION_SCALE would silently shrink the torque
    the policy can reach per unit action.
    """
    scale: dict[str, float] = {}
    for a in get_g1_identified_robot_cfg().articulation.actuators:
        assert isinstance(a, BuiltinPositionActuatorCfg)
        assert a.effort_limit is not None
        for name in a.target_names_expr:
            scale[name] = 0.25 * a.effort_limit / a.stiffness
    return scale


# Randomisation ranges, sized by how well each parameter was actually determined
# rather than by guesswork. Relative ranges multiply the nominal above.
RANDOMISATION = {
    "actuator_gain_scale": (0.95, 1.05),   # identified; tight
    "frictionloss_abs": (0.2, 2.0),        # not identifiable; wide
    "armature_rel": (0.5, 3.0),            # not identified; wide
    "viscous_damping_hips_rel": (0.7, 1.3),
    "viscous_damping_knee_ankle_abs": (0.01, 2.0),
}
