"""Generate the hardware deployment constants from the trained model.

Nothing here is hand-typed: orderings, default pose, action scales, gains and
limits are read out of the configs the policy was trained with, then written to
deploy_map.json. Asserts the orderings so a future mjlab change fails loudly.

    python -m pluto.mjlab_g1.export_deploy_map
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco as mj  # noqa: E402

import pluto.mjlab_g1 as P  # noqa: E402
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg  # noqa: E402
from mjlab.entity.entity import Entity  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

# unitree_hg motor indices, from the SDK's G1JointIndex
SDK_ORDER = [f"{n}_joint" for n in (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)]


# effort_limit -> velocity_limit. Waist/ankle double effort but not speed.
VELOCITY_LIMITS = {25.0: 37.0, 88.0: 32.0, 139.0: 20.0, 5.0: 22.0, 50.0: 37.0}


def _resolve(mapping, name, what):
    for pattern, value in mapping.items():
        if re.fullmatch(pattern, name):
            return float(value)
    raise KeyError(f"no {what} entry matches {name}")


def main() -> None:
    cfg = load_env_cfg(P.FLAT_DR_TASK)
    robot_cfg = cfg.scene.entities["robot"]
    entity = Entity(robot_cfg)
    model = entity.spec.compile()

    obs_order = list(entity.joint_names)

    # Action order follows the term's target JOINT ids, not actuator index.
    # actuator_trnid would give a wrong permutation: arm commands to leg joints.
    action_cfg = cfg.actions["joint_pos"]
    assert str(action_cfg.transmission_type).endswith("JOINT"), (
        "action transmission is not JOINT; target ids no longer index joints"
    )
    assert not action_cfg.preserve_order, (
        "preserve_order=True means the action order follows the pattern list, "
        "not the entity joint order -- re-derive it"
    )
    patterns = action_cfg.actuator_names
    if patterns is None or patterns in (".*", (".*",), [".*"]):
        action_order = list(entity.joint_names)
    else:
        pats = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        action_order = [n for n in entity.joint_names
                        if any(re.fullmatch(p, n) for p in pats)]
    assert sorted(obs_order) == sorted(SDK_ORDER) == sorted(action_order)
    assert obs_order == SDK_ORDER, (
        "observation order no longer matches the SDK motor order -- the deployment "
        "node needs an explicit permutation for joint_pos/joint_vel"
    )

    sdk_from_action = [action_order.index(n) for n in SDK_ORDER]
    assert action_order == SDK_ORDER and sdk_from_action == list(range(len(SDK_ORDER))), (
        "action order is no longer identity with the SDK motor order; the "
        "deployment node's permutation must be re-derived"
    )

    joint_pos_init = dict(robot_cfg.init_state.joint_pos)
    default_pose = [
        _resolve(joint_pos_init, n, "init joint_pos") if any(
            re.fullmatch(p, n) for p in joint_pos_init
        ) else 0.0
        for n in SDK_ORDER
    ]

    # action scale, in action order (== SDK order)
    scale = cfg.actions["joint_pos"].scale
    action_scale = [_resolve(scale, n, "action scale") for n in action_order]

    # UNCORRECTED nominal Kp. Sim uses Kp*alpha to model the torque shortfall;
    # the robot supplies alpha itself, so commanding the corrected value double-counts.
    nominal = {}
    for actuator in get_g1_robot_cfg().articulation.actuators:
        for pattern in actuator.target_names_expr:
            nominal[pattern] = (actuator.stiffness, actuator.damping)
    # armature as trained; absent from the raw XML, mjlab sets it in Python
    arm_map = {}
    for actuator in get_g1_robot_cfg().articulation.actuators:
        for pattern in actuator.target_names_expr:
            arm_map[pattern] = actuator.armature
    armature = [_resolve(arm_map, n, "armature") for n in SDK_ORDER]

    kp = [_resolve({k: v[0] for k, v in nominal.items()}, n, "kp") for n in SDK_ORDER]
    kd = [_resolve({k: v[1] for k, v in nominal.items()}, n, "kd") for n in SDK_ORDER]

    # joint limits for clipping: action 2.9 x scale 0.64 = 1.9 rad of offset
    lo, hi = [], []
    for name in SDK_ORDER:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        lo.append(float(model.jnt_range[jid][0]))
        hi.append(float(model.jnt_range[jid][1]))

    # per-joint limits: ankles 37 rad/s, knee 20; a flat threshold fits neither
    vel_lim, eff_lim = [], []
    for name in SDK_ORDER:
        act = None
        for a in get_g1_robot_cfg().articulation.actuators:
            if any(re.fullmatch(p_, name) for p_ in a.target_names_expr):
                act = a
                break
        assert act is not None, name
        eff_lim.append(float(act.effort_limit))
        vel_lim.append(float(VELOCITY_LIMITS[round(act.effort_limit, 3)]))

    actor = cfg.observations["actor"]
    out = {
        "generated_from": {
            "task": P.FLAT_DR_TASK,
            "note": "orderings and constants read from the trained config, not typed",
        },
        "policy": {
            "obs_terms": list(actor.terms),
            "history_length": getattr(actor, "history_length", None) or 1,
            "obs_dim_per_step": 3 + 3 + 29 + 29 + 29 + 3,
            "action_dim": model.nu,
            "control_hz": round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation)),
        },
        "sdk_joint_order": SDK_ORDER,
        "obs_is_identity_with_sdk": obs_order == SDK_ORDER,
        "sdk_from_action": sdk_from_action,
        "default_pose_sdk_order": default_pose,
        "action_scale_sdk_order": action_scale,
        "deploy_kp_sdk_order": kp,
        "deploy_kd_sdk_order": kd,
        "joint_lower_sdk_order": lo,
        "joint_upper_sdk_order": hi,
        "vel_limit_sdk_order": vel_lim,
        "effort_limit_sdk_order": eff_lim,
        "armature_sdk_order": armature,
    }

    path = Path(__file__).resolve().parent / "deploy_map.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(f"  obs order identity with SDK : {out['obs_is_identity_with_sdk']}")
    print(f"  action order identity       : {action_order == SDK_ORDER}")
    print(f"  control rate                : {out['policy']['control_hz']} Hz")
    print(f"  obs dim                     : "
          f"{out['policy']['obs_dim_per_step']} x {out['policy']['history_length']} = "
          f"{out['policy']['obs_dim_per_step'] * out['policy']['history_length']}")
    print(f"  deploy kp (legs, SDK 0-5)   : {[round(x, 2) for x in kp[:6]]}")


if __name__ == "__main__":
    main()
