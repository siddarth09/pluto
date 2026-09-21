"""
Synthetic dataset generator for the sim2real-robot-identification pipeline.

Plants known joint parameters in the fixed-base G1 leg model, rolls out an
excitation trajectory, and writes the .pt format the fitter expects. Because the
planted values are known, the recovered values can be scored directly.

The model is built with the pipeline's own build_fixed_base_model_xml so the
generator and the fitter simulate an identical plant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SYSID_ROOT = Path.home() / "projects25/src/sim2real-robot-identification"
sys.path.insert(0, str(SYSID_ROOT))

import mujoco as mj  # noqa: E402

import config  # noqa: E402
from sysid_mujoco.common import build_fixed_base_model_xml  # noqa: E402

ROBOT = "g1_legs"
N_JOINT = 12


def planted_params(joint_names: list[str], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Ground-truth parameters to plant. Returns armature/damping/frictionloss, each (12,).

    Nominal in g1_legs.xml is armature 0.01, damping 0.0, frictionloss 0.3 on every
    joint. Values must differ PER JOINT, otherwise a recovered value cannot be
    distinguished from a value that never moved off its initial guess.

    Armature bases are the reflected rotor inertias of the three Unitree motor types.
    The 2.2 multiplier floor is set by the smallest base (ankle, 0.0072): below it an
    ankle would be planted within 1.5x of the 0.01 nominal.
    """
    armature_by_motor = {
        "hip_pitch": 0.0102, "hip_yaw": 0.0102,        # 7520_14
        "hip_roll": 0.0251, "knee": 0.0251,            # 7520_22
        "ankle_pitch": 0.0072, "ankle_roll": 0.0072,   # 5020x2
    }

    def motor_of(name: str) -> str:
        for key in armature_by_motor:
            if key in name:
                return key
        raise KeyError(f"no motor type for {name}")

    n = len(joint_names)
    base = np.array([armature_by_motor[motor_of(j)] for j in joint_names])
    return {
        "armature": base * rng.uniform(2.2, 4.0, n),
        "damping": rng.uniform(0.05, 0.60, n),
        "frictionloss": 0.3 * rng.uniform(1.8, 3.5, n),
    }


def joint_ranges(model: mj.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(model.nu)
    hi = np.zeros(model.nu)
    for i in range(model.nu):
        j = int(model.actuator_trnid[i][0])
        lo[i], hi[i] = model.jnt_range[j]
    return lo, hi


def excitation(t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Desired joint positions over time. Returns (n_steps, 12) in radians.

    armature enters through qddot, damping through qdot, frictionloss near qdot
    sign changes -- so a single frequency cannot excite all three. A 0.2 -> 4 Hz
    chirp covers the velocity-dominated and acceleration-dominated ends; 4 Hz is
    the ceiling set by the servo bandwidth (omega_n = 10 Hz). Amplitude falls as
    1/f to hold peak velocity near v_peak: at fixed amplitude the wide-range
    joints outrun the servo and Kp*error swamps the friction and damping terms.

    Centred on each joint's range midpoint, not on q0: the knee range is
    [-0.087, 2.880], so a swing about zero would clip on ctrlrange and flatten
    the velocity on the highest-inertia joint.
    """
    f0, f1 = 0.2, 4.0
    v_peak = 3.0
    span = t[-1]
    centre = 0.5 * (lo + hi)
    # peak velocity is amp * 2*pi*f, so hold it fixed by sweeping amp as 1/f;
    # at the low-frequency end the range cap binds instead
    f = f0 + (f1 - f0) * t / span
    amp = np.minimum(
        (v_peak / (2.0 * np.pi * f))[:, None],
        (0.4 * 0.5 * (hi - lo))[None, :],
    )
    # phase is the integral of f(t); sin(2*pi*f(t)*t) sweeps to twice f1
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / span)
    # per-joint offsets decorrelate the regressor columns
    offs = np.linspace(0.0, 2.0 * np.pi, len(lo), endpoint=False)
    return centre[None, :] + amp * np.sin(phase[:, None] + offs[None, :])


def build_model() -> mj.MjModel:
    gains = {
        name: (float(kp), float(kd))
        for name, kp, kd in zip(joint_names_from_source(), config.Kp, config.Kd)
    }
    xml = build_fixed_base_model_xml(ROBOT, actuator_mode="general", actuator_gains=gains)
    return mj.MjModel.from_xml_path(str(xml))


def joint_names_from_source() -> list[str]:
    m = mj.MjModel.from_xml_path(str(SYSID_ROOT / "robot_model" / ROBOT / f"{ROBOT}.xml"))
    return [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i][0]))
        for i in range(m.nu)
    ]


def apply_params(model: mj.MjModel, p: dict[str, np.ndarray]) -> None:
    for i in range(model.nu):
        dof = int(model.jnt_dofadr[int(model.actuator_trnid[i][0])])
        model.dof_armature[dof] = p["armature"][i]
        model.dof_damping[dof] = p["damping"][i]
        model.dof_frictionloss[dof] = p["frictionloss"][i]


def rollout(model: mj.MjModel, q_des: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Roll out the excitation, matching eval_fit.simulate_open_loop's convention.

    State k is recorded BEFORE control k is applied, so measurement k and control k
    share timestamp t[k]. Recording after the step instead shifts the whole record
    by one timestep relative to how the fitter replays it, which puts a floor under
    the residual that no parameter value can remove.
    """
    data = mj.MjData(model)
    # start on the trajectory: qpos defaults to zero but the excitation is centred
    # on each joint's range midpoint, which would be a step into a stiff servo
    data.qpos[: model.nu] = q_des[0]
    mj.mj_forward(model, data)
    n = len(q_des)
    q = np.zeros((n, model.nu))
    dq = np.zeros((n, model.nu))
    q[0] = data.qpos[: model.nu]
    dq[0] = data.qvel[: model.nu]
    for k in range(n - 1):
        data.ctrl[:] = q_des[k]
        mj.mj_step(model, data)
        q[k + 1] = data.qpos[: model.nu]
        dq[k + 1] = data.qvel[: model.nu]
    return {"dof_pos": q, "dof_vel": dq}


def main() -> None:
    dt = 1.0 / config.frequency_collection
    model = build_model()
    model.opt.timestep = dt
    assert model.nu == N_JOINT, f"expected {N_JOINT} actuators, got {model.nu}"

    names = joint_names_from_source()
    rng = np.random.default_rng(0)
    truth = planted_params(names, rng)
    apply_params(model, truth)

    lo, hi = joint_ranges(model)
    t = np.arange(0.0, 20.0, dt)
    q_des = excitation(t, lo, hi)
    assert q_des.shape == (len(t), N_JOINT), f"excitation shape {q_des.shape}"

    out = rollout(model, q_des, dt)

    dest = SYSID_ROOT / "datasets" / ROBOT
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": torch.as_tensor(t, dtype=torch.float64),
            "dof_pos": torch.as_tensor(out["dof_pos"], dtype=torch.float32),
            "dof_vel": torch.as_tensor(out["dof_vel"], dtype=torch.float32),
            "des_dof_pos": torch.as_tensor(q_des, dtype=torch.float32),
            "des_dof_vel": torch.zeros_like(torch.as_tensor(q_des, dtype=torch.float32)),
            "kp": torch.as_tensor(config.Kp, dtype=torch.float32),
            "kd": torch.as_tensor(config.Kd, dtype=torch.float32),
        },
        dest / "traj_0.pt",
    )
    results = Path(__file__).resolve().parent.parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez(results / "planted_truth.npz", names=names, **truth)
    print(f"wrote {dest / 'traj_0.pt'}  ({len(t)} samples at {config.frequency_collection} Hz)")
    print(f"answer key -> results/planted_truth.npz")


if __name__ == "__main__":
    main()
