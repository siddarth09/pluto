"""A MuJoCo stand-in for the robot, so deploy.py can run without hardware.

Exposes the same `state` dict and `publish` function as the DDS path, so the
deployment node's observation assembly, action mapping, teleop and safety logic
run unchanged.

UNVERIFIED: this has not been run end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco as mj
import numpy as np

G1_XML = Path("/home/sid/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml")
PHYSICS_DT = 0.005


class SimRobot:
    def __init__(self, deploy_map: dict, identified: dict | None = None):
        self.names = deploy_map["sdk_joint_order"]
        kp = np.array(deploy_map["deploy_kp_sdk_order"])
        kd = np.array(deploy_map["deploy_kd_sdk_order"])
        default = np.array(deploy_map["default_pose_sdk_order"])

        spec = mj.MjSpec.from_file(str(G1_XML))
        spec.option.timestep = PHYSICS_DT
        # without implicitfast a Kp=99 servo at dt=5 ms is unstable
        spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

        # floor; the asset zoo XML is the robot only
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mj.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.rgba = [0.3, 0.3, 0.35, 1.0]

        # position actuators in SDK order, so ctrl[i] == motor_cmd[i]
        for i, name in enumerate(self.names):
            a = spec.add_actuator()
            a.name = f"{name}_pos"
            a.target = name
            a.trntype = mj.mjtTrn.mjTRN_JOINT
            a.gaintype = mj.mjtGain.mjGAIN_FIXED
            a.biastype = mj.mjtBias.mjBIAS_AFFINE
            a.gainprm[0] = kp[i]
            a.biasprm[1] = -kp[i]
            a.biasprm[2] = -kd[i]

        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        # armature as trained; its absence is what blows up the stiff joints
        for i, name in enumerate(self.names):
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.model.dof_armature[self.model.jnt_dofadr[jid]] = \
                    deploy_map["armature_sdk_order"][i]

        # identified leg parameters
        if identified:
            for key, arr in (("armature_fitted", self.model.dof_armature),
                             ("damping_fitted", self.model.dof_damping),
                             ("frictionloss_measured", self.model.dof_frictionloss)):
                for jname, val in identified.get(key, {}).items():
                    jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        arr[self.model.jnt_dofadr[jid]] = val

        self.nq_root = self.model.nq - len(self.names)
        self.reset(default)

    def reset(self, default: np.ndarray) -> None:
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.nq_root:] = default
        if self.nq_root == 7:
            self.data.qpos[:3] = [0.0, 0.0, 0.80]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(self.model, self.data)
        # settle onto the floor
        for _ in range(int(0.5 / PHYSICS_DT)):
            self.data.ctrl[:] = default
            mj.mj_step(self.model, self.data)

    def read(self) -> dict:
        d, n = self.data, len(self.names)
        quat = d.qpos[3:7] if self.nq_root == 7 else np.array([1.0, 0, 0, 0])
        # freejoint qvel[3:6] is already body-frame
        gyro = d.qvel[3:6] if self.nq_root == 7 else np.zeros(3)
        return {
            "q": d.qpos[self.nq_root:].copy(),
            "dq": d.qvel[self.nq_root - 1:].copy()[:n] if self.nq_root == 7
            else d.qvel[:n].copy(),
            "tau": d.actuator_force[:n].copy(),
            "gyro": np.asarray(gyro).copy(),
            "quat": np.asarray(quat).copy(),
            "mode_machine": 0,
        }

    def step(self, q_des: np.ndarray, gain: float, n_sub: int) -> None:
        # `gain` scales the servo the way ramping does on hardware; MuJoCo has no
        # per-step gain input, so fold it into the target by blending toward the
        # measured position (gain 0 -> no correction, i.e. limp).
        q_now = self.data.qpos[self.nq_root:]
        target = q_now + gain * (q_des - q_now)
        for _ in range(n_sub):
            self.data.ctrl[:] = target
            mj.mj_step(self.model, self.data)


def load(map_path: Path, identified_path: Path | None = None):
    dm = json.loads(Path(map_path).read_text())
    ident = None
    if identified_path and Path(identified_path).is_file():
        ident = json.loads(Path(identified_path).read_text())
    return SimRobot(dm, ident)
