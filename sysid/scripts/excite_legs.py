"""Commanded leg excitation on the real G1, recorded in the sysid dataset format.

Releases the motion-control service, runs a stepped-frequency excitation under a
soft position servo, writes the 7-key .pt.

Only safe with the robot hoisted and both feet clear. Sequence: hold current pose
-> ramp gains in -> ramp to trajectory start -> excite -> ramp back -> gains out.
The watchdog zeroes gains on excess error or velocity.

    python scripts/excite_legs.py --dry-run
    python scripts/excite_legs.py --only left_knee --centre 1.0 --amp 0.8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/sid/unitree_sdk2_python")
sys.path.insert(0, "/home/sid/projects25/src/sim2real-robot-identification")

import config  # noqa: E402
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,  # noqa: E402
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except ImportError:
    from unitree_sdk2py.g1.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

N_LEG = 12
HZ = 200.0
DT = 1.0 / HZ
# Joint limits, actuator order: L hip_p/r/y knee ank_p/r then R.
LIMIT_LO = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                     -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618])
LIMIT_HI = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                     2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618])
# static friction lower bounds: gravity torque each joint held without moving.
# Roll joints were gravity-neutral, so they inherit the ankle figure.
STICTION = np.array([0.53, 0.59, 0.16, 0.85, 0.21, 0.21,
                     0.53, 0.59, 0.16, 0.85, 0.21, 0.21])
# Gravity-neutral hanging pose, not the joint range midpoint: midpoint puts
# hip_roll at 70 deg abduction, ~25 N m to hold. hip_roll offset outward per leg.
NEUTRAL = np.array([0.0, 0.20, 0.0, 0.45, -0.15, 0.0,
                    0.0, -0.20, 0.0, 0.45, -0.15, 0.0])
# Amplitude ceiling from the gravity budget, not joint limits: holding at A
# costs ~C*sin(A) and the servo pays C*sin(A)/kp in droop. hip_pitch binds.
AMP_MAX = np.array([0.30, 0.45, 0.60, 0.35, 0.45, 0.13,
                    0.30, 0.45, 0.60, 0.35, 0.45, 0.13])
NAMES = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
         "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
         "right_hip_roll", "right_hip_yaw", "right_knee",
         "right_ankle_pitch", "right_ankle_roll"]

MAX_TRACK_ERR = 0.60   # rad, overridable; see --max-err
MAX_VEL = 8.0          # rad/s


# Stepped frequencies, not a continuous chirp: 1/f amplitude leaves acceleration
# growing only as f, and the top of the sweep falls below the stiction threshold.
# (freq Hz, duration s, fixed amplitude rad or None for velocity-capped)
SEGMENTS = ((0.5, 12.0, None),    # damping, frictionloss: slow, large, many reversals
            (1.0, 6.0, None),
            (2.0, 6.0, "hf"),     # armature: acceleration scales as f^2
            (4.0, 6.0, "hf"))


def segmented(lo, hi, dt, v_cap, hf_amp, amp_scale, taper=0.5, only=None,
              centre_override=None, amp_override=None):
    """Stepped-frequency excitation. Returns (n, 12) positions and a per-segment table."""
    centre = NEUTRAL.copy()
    amp_max = AMP_MAX.copy()
    # single-joint runs: only this joint's limits and droop constrain it
    if only is not None and centre_override is not None:
        centre[only] = centre_override
    if only is not None and amp_override is not None:
        amp_max[only] = amp_override
    # usable swing is distance to the nearest limit, not half the range
    half = np.minimum(np.minimum(centre - lo, hi - centre) - 0.10, amp_max)
    offs = np.linspace(0.0, 2.0 * np.pi, len(lo), endpoint=False)
    mask = np.ones(len(lo)) if only is None else np.zeros(len(lo))
    if only is not None:
        mask[only] = 1.0
    blocks, table = [], []
    for freq, dur, mode in SEGMENTS:
        n = int(round(dur / dt))
        ts = np.arange(n) * dt
        w = 2.0 * np.pi * freq
        if mode == "hf":
            amp = np.minimum(np.full_like(half, hf_amp), half)
        else:
            amp = np.minimum(half, v_cap / w)
        amp = amp * amp_scale * mask
        # taper to zero at both ends so every segment boundary lands on `centre`
        env = np.ones(n)
        k = max(1, int(round(taper / dt)))
        ramp = smoothstep(k)
        env[:k] = ramp
        env[-k:] = ramp[::-1]
        blocks.append(centre[None, :]
                      + (env[:, None] * amp[None, :]) * np.sin(w * ts[:, None] + offs[None, :]))
        table.append((freq, dur, amp, amp * w, amp * w * w, int(2 * freq * dur)))
    return np.concatenate(blocks), table


def smoothstep(n):
    """Min-jerk 0->1 over n samples: zero velocity and acceleration at both ends."""
    s = np.linspace(0.0, 1.0, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--v-cap", type=float, default=3.0,
                    help="peak velocity cap for the low-frequency segments, rad/s")
    ap.add_argument("--hf-amp", type=float, default=0.13,
                    help="fixed amplitude for the high-frequency segments, rad")
    ap.add_argument("--kp-scale", type=float, default=0.5,
                    help="fraction of config.Kp; Kd scales as sqrt to hold damping ratio")
    ap.add_argument("--amp-scale", type=float, default=1.0, help="extra amplitude derate")
    ap.add_argument("--ramp", type=float, default=3.0, help="seconds for each ramp phase")
    ap.add_argument("--centre", type=float, default=None,
                    help="With --only: move that joint's oscillation centre away "
                         "from NEUTRAL, to open up swing against its limits.")
    ap.add_argument("--amp", type=float, default=None,
                    help="With --only: raise that joint's amplitude ceiling above "
                         "AMP_MAX (still clipped by limits and --v-cap).")
    ap.add_argument("--only", default=None,
                    help="Excite ONE joint, holding the rest at NEUTRAL. Name "
                         "(e.g. left_knee) or index 0-11. Decouples the residual "
                         "so a per-joint fit sees only that joint's dynamics.")
    ap.add_argument("--max-err", type=float, default=MAX_TRACK_ERR,
                    help="watchdog tracking-error limit, rad. Gravity droop is "
                         "expected and predictable; velocity is the runaway guard.")
    ap.add_argument("--iface", default="enp130s0")
    ap.add_argument("--out", default="chirp_0.pt")
    ap.add_argument("--dir", default="/home/sid/projects25/src/pluto/sysid/data")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check the trajectory; never release or publish")
    args = ap.parse_args()

    kp = config.Kp[:N_LEG] * args.kp_scale
    kd = config.Kd[:N_LEG] * np.sqrt(args.kp_scale)

    only = None
    if args.only is not None:
        if args.only.isdigit():
            only = int(args.only)
        else:
            matches = [i for i, nm in enumerate(NAMES) if args.only in nm]
            if len(matches) != 1:
                raise SystemExit(f"--only {args.only!r} matched {len(matches)}: "
                                 f"{[NAMES[i] for i in matches]}")
            only = matches[0]
        print(f"single-joint mode: {NAMES[only]} (index {only}); others held at NEUTRAL")
    q_ex, table = segmented(LIMIT_LO, LIMIT_HI, DT, args.v_cap, args.hf_amp,
                            args.amp_scale, only=only,
                            centre_override=args.centre, amp_override=args.amp)

    margin = 0.10
    ok = bool((q_ex > LIMIT_LO + margin).all() and (q_ex < LIMIT_HI - margin).all())
    print(f"trajectory {q_ex.shape}  {q_ex.shape[0] * DT:.1f}s  "
          f"within limits (>{margin} rad margin): {ok}")
    print(f"{'seg':>4}{'Hz':>6}{'s':>5}{'amp_knee':>10}{'v_pk':>7}{'a_pk':>8}{'revs':>6}")
    for i, (f, dur, amp, v, a, rev) in enumerate(table):
        print(f"{i:>4}{f:>6.1f}{dur:>5.0f}{amp[3]:>10.3f}{v[3]:>7.2f}{a[3]:>8.1f}{rev:>6d}")
    print(f"  total reversals per joint {sum(r for *_, r in table)}")
    print(f"  amplitude span {(q_ex.max(0) - q_ex.min(0)).round(3)}")
    print(f"  kp {kp.round(1)}")
    print(f"  kd {kd.round(2)}")
    # below the breakaway error you get chatter, not motion
    stiction_err = STICTION / kp
    hf = args.hf_amp * args.amp_scale
    print(f"  breakaway err (measured stiction / kp) {stiction_err.round(3)} rad")
    print(f"  hf amplitude {hf:.3f} rad = {hf / stiction_err.max():.1f}x the worst "
          f"breakaway -> {'OK' if hf > 3 * stiction_err.max() else 'TOO SMALL'}")
    if not ok:
        raise SystemExit("trajectory exceeds joint limits -- lower --amp-scale or --hf-amp")

    ChannelFactoryInitialize(0, args.iface)
    state = {}
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: state.update(
            q=np.array([x.q for x in m.motor_state[:N_LEG]]),
            dq=np.array([x.dq for x in m.motor_state[:N_LEG]]),
            tau=np.array([x.tau_est for x in m.motor_state[:N_LEG]]),
            temp=np.array([x.temperature[0] for x in m.motor_state[:N_LEG]]),
            mode_machine=m.mode_machine), 20)
    t0 = time.time()
    while "q" not in state and time.time() - t0 < 5.0:
        time.sleep(0.02)
    if "q" not in state:
        raise SystemExit("no lowstate -- check --iface")
    q_start = state["q"].copy()
    print(f"\ncurrent q {q_start.round(3)}")
    print(f"mode_machine {state['mode_machine']}")

    if args.dry_run:
        print("\ndry run: nothing released, nothing published")
        return

    print("\nThis RELEASES the motion service and COMMANDS the legs.")
    print("Both feet must be clear of the floor. Keep a hand on the stop.")
    if input("Type EXCITE to proceed: ").strip() != "EXCITE":
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

    n_ramp = int(args.ramp / DT)
    # gains in -> move to q_ex[0] -> excite -> back -> gains out
    plan = []
    plan += [(q_start, g) for g in np.linspace(0.0, 1.0, n_ramp)]
    for a in smoothstep(n_ramp):
        plan.append((q_start + a * (q_ex[0] - q_start), 1.0))
    plan += [(q, 1.0) for q in q_ex]
    for a in smoothstep(n_ramp):
        plan.append((q_ex[-1] + a * (q_start - q_ex[-1]), 1.0))
    plan += [(q_start, g) for g in np.linspace(1.0, 0.0, n_ramp)]
    chirp_lo = 2 * n_ramp
    chirp_hi = chirp_lo + len(q_ex)

    rec_q, rec_dq, rec_des, rec_tau, rec_temp = [], [], [], [], []
    aborted = None
    next_t = time.time()
    for k, (q_des, gain) in enumerate(plan):
        # record state BEFORE the command: state k pairs with control k
        rec_q.append(state["q"].copy()); rec_dq.append(state["dq"].copy())
        rec_des.append(q_des.copy()); rec_tau.append(state["tau"].copy())
        rec_temp.append(state["temp"].copy())

        err = np.abs(state["q"] - q_des)
        if gain > 0.99 and (err.max() > args.max_err or np.abs(state["dq"]).max() > MAX_VEL):
            aborted = (f"watchdog at k={k}: max|err|={err.max():.3f} "
                       f"max|dq|={np.abs(state['dq']).max():.2f}")
            break

        for j in range(N_LEG):
            mc = cmd.motor_cmd[j]
            mc.mode = 1
            mc.q = float(np.clip(q_des[j], LIMIT_LO[j] + 0.05, LIMIT_HI[j] - 0.05))
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(kp[j] * gain)
            mc.kd = float(kd[j] * gain)
        for j in range(N_LEG, 35):          # everything above the legs stays limp
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        next_t += DT
        sleep = next_t - time.time()
        if sleep > 0:
            time.sleep(sleep)

    # always leave the joints limp
    for _ in range(100):
        for j in range(35):
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(DT)
    print("\ngains zeroed, joints limp")
    if aborted:
        print("ABORTED:", aborted)

    q = np.stack(rec_q); dq = np.stack(rec_dq); des = np.stack(rec_des)
    tau = np.stack(rec_tau); temp = np.stack(rec_temp)
    sl = slice(chirp_lo, min(chirp_hi, len(q)))
    if sl.stop - sl.start < 100:
        raise SystemExit("chirp phase too short to save -- see abort reason above")
    t = np.arange(sl.stop - sl.start) * DT

    out = Path(args.dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": torch.as_tensor(t, dtype=torch.float64),
        "dof_pos": torch.as_tensor(q[sl], dtype=torch.float32),
        "dof_vel": torch.as_tensor(dq[sl], dtype=torch.float32),
        "des_dof_pos": torch.as_tensor(des[sl], dtype=torch.float32),
        "des_dof_vel": torch.zeros_like(torch.as_tensor(des[sl], dtype=torch.float32)),
        "kp": torch.as_tensor(kp, dtype=torch.float32),
        "kd": torch.as_tensor(kd, dtype=torch.float32),
    }, out / args.out)
    np.savez(out / (Path(args.out).stem + "_meta.npz"),
             joint_names=NAMES, tau_est=tau[sl], temperature=temp[sl],
             full_q=q, full_des=des, kp=kp, kd=kd)

    e = np.abs(des[sl] - q[sl])
    print(f"\nwrote {out / args.out}  ({sl.stop - sl.start} samples at {HZ} Hz)")
    print(f"peak |dq|   {np.abs(dq[sl]).max(0).round(2)}")
    print(f"rom         {(q[sl].max(0) - q[sl].min(0)).round(3)}")
    print(f"track err   mean {e.mean():.4f}  max {e.max():.3f}")
    print(f"peak |tau|  {np.abs(tau[sl]).max(0).round(1)}")
    print(f"temperature {temp[sl].mean(0).round(0)}")


if __name__ == "__main__":
    main()
