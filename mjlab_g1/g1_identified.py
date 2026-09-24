

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

# Measured on hardware 2026-09-21. gain_scale: tau_est / commanded PD torque.
# frictionloss: static breakaway, not fitted. armature: motor spec, not fitted.
# Identifiability verdicts in sysid/results/g1_legs_identified.json.
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
    # 4-bar parallel linkage, two 5020s: mjlab doubles armature/stiffness/effort.
    # gain_scale left at 1.0, per-motor tau_est does not map through the linkage.
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


def _leg_actuators(delay_max_lag: int = 0, delay_min_lag: int = 0) -> tuple[BuiltinPositionActuatorCfg, ...]:
    out = []
    for joint, p in _LEG.items():
        for side_index, side in enumerate(("left", "right")):
            g = p["gain"][side_index]
            out.append(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(f"{side}_{joint}_joint",),
                    # scale the sim servo to the torque the robot actually delivers
                    stiffness=p["stiff"] * g,
                    damping=p["kd"] * g,
                    effort_limit=p["effort"],
                    armature=p["arm"],
                    frictionloss=p["friction"][side_index],
                    viscous_damping=p["damp"][side_index],
                    # command lag in physics steps (5 ms each, 4 per policy step)
                    delay_min_lag=delay_min_lag,
                    delay_max_lag=delay_max_lag,
                    # lag persists rather than resampling every step
                    delay_hold_prob=0.9 if delay_max_lag else 0.0,
                )
            )
    return tuple(out)


def _waist_yaw_only() -> BuiltinPositionActuatorCfg:
    """Upstream groups waist_yaw with the hip joints; the hips are now per-joint."""
    return dataclasses.replace(
        G1_ACTUATOR_7520_14, target_names_expr=("waist_yaw_joint",)
    )


def get_g1_identified_robot_cfg(
    delay_max_lag: int = 0, delay_min_lag: int = 0
) -> EntityCfg:
    """Identified G1. Lags are in physics steps (5 ms each); 0 disables.

    Setting min == max gives a DETERMINISTIC delay, for reproducing a measured
    hardware latency rather than randomising over a range.
    """
    cfg = get_g1_robot_cfg()
    actuators = (
        G1_ACTUATOR_5020,     # arms, unchanged
        G1_ACTUATOR_4010,     # wrists, unchanged
        G1_ACTUATOR_WAIST,    # waist pitch/roll, unchanged
        _waist_yaw_only(),
        *_leg_actuators(delay_max_lag=delay_max_lag, delay_min_lag=delay_min_lag),
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

RANDOMISATION = {
    "actuator_gain_scale": (0.95, 1.05),   # identified; tight
    "frictionloss_abs": (0.2, 2.0),        # not identifiable; wide
    "armature_rel": (0.5, 3.0),            # not identified; wide
    "viscous_damping_hips_rel": (0.7, 1.3),
    "viscous_damping_knee_ankle_abs": (0.01, 2.0),
    "actuator_delay_steps": (0, 4),  # 0-20 ms; one 50 Hz policy step
}
