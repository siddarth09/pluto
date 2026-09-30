"""Measure arm static breakaway torque on the real G1.

The fit cannot determine Coulomb friction -- it spans orders of magnitude for
almost no change in predictive accuracy -- so measure it directly. Ramp torque
into one joint and record the value at which it starts to move. Below breakaway
the joint chatters, it does not travel.

One joint at a time, the rest held at NEUTRAL. Only safe with the robot hoisted.

    python scripts/stiction_arms.py --dry-run
    python scripts/stiction_arms.py --only left_elbow
    python scripts/stiction_arms.py --out arm_stiction.npz
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, "/home/sid/unitree_sdk2_python")

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,  # noqa: E402
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except ImportError:
    from unitree_sdk2py.g1.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

N_MOTOR = 35
ARM_IDX = np.arange(15, 29)
WAIST_IDX = np.arange(12, 15)
HZ = 200.0
DT = 1.0 / HZ

NAMES = [f"{s}_{j}" for s in ("left", "right") for j in
         ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_roll", "wrist_pitch", "wrist_yaw")]
EFFORT = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                   25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])
WAIST_KP = np.array([28.50, 28.50, 28.50])
WAIST_KD = np.array([1.814, 1.814, 1.814])

# Gravity-neutral hanging pose, matching excite_arms.py. The joint under test is
# released from here and the rest are held.
NEUTRAL = np.array([0.0, 0.20, 0.0, 1.10, 0.0, 0.0, 0.0,
                    0.0, -0.20, 0.0, 1.10, 0.0, 0.0, 0.0])
ARM_KP = np.array([14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78,
                   14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78])
ARM_KD = np.array([0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
                   0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068])

# The released joint is gravity-compensated so the ramp measures friction alone.
# Gravity at the shoulders is an order of magnitude above their friction, so
# uncompensated the arm just falls. Compensation is evaluated at the pose the arm
# settles at, not at nominal NEUTRAL: the servo droops under gravity and
# dtau_g/dq is steep at shoulder_roll.
ARMS_XML = ("/home/sid/projects25/src/sim2real-robot-identification/"
            "sysid_mujoco/generated/g1_arms/g1_arms_fixed_base_sysid.xml")


class Gravity:
    """qfrc_bias for the 14 arm joints at a measured pose."""

    def __init__(self, path=ARMS_XML):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        if self.m.nq != len(NAMES):
            raise SystemExit(f"{path} has nq={self.m.nq}, expected {len(NAMES)}")

    def at(self, q):
        self.d.qpos[:] = q
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return self.d.qfrc_bias.copy()

RAMP_S = 4.0        # seconds to ramp from zero to the torque ceiling
# The ceiling only has to cover friction, so keep it small: finer resolution.
CEIL_FRAC = 0.15    # ceiling as a fraction of spec torque
MOVE_THRESH = 0.02  # rad; travel that counts as broken away
SETTLE_S = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--only", default=None, help="one joint by name or index 0-13")
    ap.add_argument("--ramp", type=float, default=RAMP_S)
    ap.add_argument("--ceil-frac", type=float, default=CEIL_FRAC)
    ap.add_argument("--iface", default="enp130s0")
    ap.add_argument("--out", default="arm_stiction.npz")
    ap.add_argument("--dir", default="/home/sid/projects25/src/pluto/sysid/results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = list(range(len(NAMES)))
    if args.only is not None:
        if args.only.isdigit():
            targets = [int(args.only)]
        else:
            m = [i for i, nm in enumerate(NAMES) if args.only in nm]
            if len(m) != 1:
                raise SystemExit(f"--only {args.only!r} matched {[NAMES[i] for i in m]}")
            targets = m

    ceil = EFFORT * args.ceil_frac
    print(f"joints: {[NAMES[i] for i in targets]}")
    print(f"torque ceiling {ceil.round(2)} N m ({args.ceil_frac:.0%} of spec)")
    print(f"ramp {args.ramp:.1f}s, breakaway threshold {MOVE_THRESH} rad")
    if args.dry_run:
        print("\ndry run: nothing released, nothing published")
        return

    ChannelFactoryInitialize(0, args.iface)
    state = {}
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: state.update(
            q=np.array([x.q for x in m.motor_state])[ARM_IDX],
            dq=np.array([x.dq for x in m.motor_state])[ARM_IDX],
            tau=np.array([x.tau_est for x in m.motor_state])[ARM_IDX],
            waist_q=np.array([x.q for x in m.motor_state])[WAIST_IDX],
            mode_machine=m.mode_machine), 20)
    t0 = time.time()
    while "q" not in state and time.time() - t0 < 5.0:
        time.sleep(0.02)
    if "q" not in state:
        raise SystemExit("no lowstate -- check --iface")

    print("\nThis RELEASES the motion service and applies raw torque to the arms.")
    print("The robot must be hoisted with both arms swinging clear.")
    if input("Type BREAKAWAY to proceed: ").strip() != "BREAKAWAY":
        print("aborted")
        return

    ms = MotionSwitcherClient(); ms.SetTimeout(5.0); ms.Init()
    print("ReleaseMode ->", ms.ReleaseMode())
    time.sleep(1.0)
    print("CheckMode   ->", ms.CheckMode())

    pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
    crc = CRC()
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = state["mode_machine"]
    waist_hold = state["waist_q"].copy()

    grav = Gravity()
    tau_g = np.zeros(len(NAMES))

    def publish(tau_arm, free=None, gain=1.0):
        """Hold every arm joint at NEUTRAL except `free`, which gets raw torque."""
        for n, j in enumerate(ARM_IDX):
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.dq = 0.0
            if free is not None and n == free:
                # gravity feedforward plus the ramp, no position feedback.
                # Clamped: an overestimate would push the joint away and keep
                # pushing.
                t = np.clip(tau_g[n] + tau_arm[n], -0.5 * EFFORT[n], 0.5 * EFFORT[n])
                mc.q = 0.0; mc.tau = float(t)
                mc.kp = 0.0; mc.kd = 0.0
            else:
                mc.q = float(NEUTRAL[n]); mc.tau = 0.0
                mc.kp = float(ARM_KP[n] * gain); mc.kd = float(ARM_KD[n] * gain)
        for n, j in enumerate(WAIST_IDX):       # keep the pelvis rigid
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.q = float(waist_hold[n]); mc.dq = 0.0; mc.tau = 0.0
            mc.kp = float(WAIST_KP[n]); mc.kd = float(WAIST_KD[n])
        for j in list(range(0, 12)) + list(range(29, N_MOTOR)):
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    # With gravity fed forward both directions return the same mu; the
    # half-difference is the compensation error.
    breakaway = np.full((len(NAMES), 2), np.nan)
    traces = {}
    n_ramp = int(args.ramp / DT)
    n_settle = int(SETTLE_S / DT)
    try:
        # ramp gains in and move to NEUTRAL, so gravity at breakaway is known
        for g in np.linspace(0.0, 1.0, n_settle):
            publish(np.zeros(len(NAMES)), gain=float(g))
            time.sleep(DT)
        for _ in range(int(2.0 / DT)):
            publish(np.zeros(len(NAMES)))
            time.sleep(DT)
        print(f"arms at NEUTRAL, err {np.abs(state['q'] - NEUTRAL).round(3)}")
        print(f"gravity at the settled pose {grav.at(state['q']).round(3)}")
        for idx in targets:
            for sign in (+1.0, -1.0):
                # Settle on gravity feedforward alone before ramping, and
                # re-evaluate it from live encoders every sample: dtau_g/dq is
                # steep enough that drift during the ramp is the size of mu.
                for _ in range(int(0.5 / DT)):
                    tau_g = grav.at(state["q"])
                    publish(np.zeros(len(NAMES)), free=idx)
                    time.sleep(DT)
                q0 = state["q"][idx]
                rec = []
                hit = None
                for k in range(n_ramp):
                    tau_g = grav.at(state["q"])
                    tau_arm = np.zeros(len(NAMES))
                    tau_arm[idx] = sign * ceil[idx] * (k / n_ramp)
                    publish(tau_arm, free=idx)
                    rec.append((tau_arm[idx], state["q"][idx], state["tau"][idx]))
                    if abs(state["q"][idx] - q0) > MOVE_THRESH:
                        # commanded ramp, not tau_est: gravity is already fed
                        # forward, so this is the friction contribution
                        hit = abs(tau_arm[idx])
                        break          # done, stop pushing
                    time.sleep(DT)
                for _ in range(n_settle):   # back to NEUTRAL before the next direction
                    publish(np.zeros(len(NAMES)))
                    time.sleep(DT)
                traces[f"{NAMES[idx]}_{'pos' if sign > 0 else 'neg'}"] = np.array(rec)
                tag = "+" if sign > 0 else "-"
                col = 0 if sign > 0 else 1
                if hit is None:
                    print(f"  {NAMES[idx]:22s} {tag} no breakaway below "
                          f"{ceil[idx]:.2f} N m -- raise --ceil-frac")
                else:
                    breakaway[idx, col] = hit
                    print(f"  {NAMES[idx]:22s} {tag} breakaway {hit:.3f} N m")
            both = breakaway[idx]
            if not np.isnan(both).any():
                print(f"  {NAMES[idx]:22s}   -> mu {both.mean():.3f} N m, "
                      f"gravity-comp error {abs(both[1] - both[0]) / 2:.3f} N m")
    finally:
        for _ in range(100):
            publish(np.zeros(len(NAMES)))
            time.sleep(DT)
        print("\ngains and torques zeroed, joints limp")

    mu = breakaway.mean(axis=1)
    tau_g = np.abs(breakaway[:, 1] - breakaway[:, 0]) / 2.0   # comp error now
    out = Path(args.dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / args.out, joint_names=NAMES, breakaway=breakaway,
             stiction=mu, implied_tau_g=tau_g, ceiling=ceil, **traces)
    print(f"\nwrote {out / args.out}")
    print("paste into excite_arms.py:")
    print("STICTION = np.array([" + ", ".join(
        "np.nan" if np.isnan(v) else f"{v:.2f}" for v in mu) + "])")
    print("gravity-comp error (should be small; large means TAU_G is wrong "
          "at the settled pose): " + np.array2string(tau_g, precision=2))


if __name__ == "__main__":
    main()
