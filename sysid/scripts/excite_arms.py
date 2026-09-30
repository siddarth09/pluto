"""Commanded arm excitation on the real G1, recorded in the sysid dataset format.

Same method as excite_legs.py, different mounting. The harness holds the torso
through the shoulders, so the torso is the fixed body and the arms hang free.
The waist is held at FULL gain, not limp: that is what makes the pelvis and the
legs rigid with the harnessed torso, which is the assumption g1_arms.xml is
built on. The legs stay limp and hang.

Only safe with the robot hoisted. Sequence: hold current pose -> ramp gains in
-> ramp to trajectory start -> excite -> ramp back -> gains out. The watchdog
zeroes gains on excess error or velocity.

    python scripts/excite_arms.py --dry-run
    python scripts/excite_arms.py --only left_shoulder_pitch --kp-scale 1.0
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

N_ARM = 14
N_MOTOR = 35
ARM_IDX = np.arange(15, 29)       # L shoulder p/r/y, elbow, wrist r/p/y, then R
WAIST_IDX = np.arange(12, 15)     # yaw, roll, pitch: held rigid, not fitted
LEG_IDX = np.arange(0, 12)        # limp, hanging
HZ = 200.0
DT = 1.0 / HZ

NAMES = [f"{s}_{j}" for s in ("left", "right") for j in
         ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_roll", "wrist_pitch", "wrist_yaw")]

# Joint limits from g1_arms.xml. shoulder_roll is mirrored per side.
LIMIT_LO = np.array([-3.0892, -1.5882, -2.6180, -1.0472, -1.97222, -1.61443, -1.61443,
                     -3.0892, -2.2515, -2.6180, -1.0472, -1.97222, -1.61443, -1.61443])
LIMIT_HI = np.array([2.6704, 2.2515, 2.6180, 2.0944, 1.97222, 1.61443, 1.61443,
                     2.6704, 1.5882, 2.6180, 2.0944, 1.97222, 1.61443, 1.61443])
# Per-joint spec torque: 5020 for shoulders/elbow/wrist_roll, 4010 for wrist p/y.
EFFORT = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                   25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])
# Measured 2026-09-28 by stiction_arms.py with gravity fed forward from live
# encoder positions. Six of seven mirrored pairs agree within 21%. shoulder_roll
# does not and is unresolved: 2.19 left, over the 3.75 ceiling right.
STICTION = np.array([0.65, 2.19, 0.50, 0.48, 0.46, 0.27, 0.22,
                     0.60, 1.81, 0.52, 0.61, 0.46, 0.31, 0.20])
# Gravity-neutral hanging pose, not the joint midpoint. qpos=0 puts 3.64 N m on
# shoulder_pitch, 0.26 rad of sag at kp=14.25 before any motion is commanded.
NEUTRAL = np.array([0.0, 0.20, 0.0, 1.10, 0.0, 0.0, 0.0,
                    0.0, -0.20, 0.0, 1.10, 0.0, 0.0, 0.0])
# Shoulder pitch and roll are droop-bound at 0.35 rad; the rest bind on limits.
AMP_MAX = np.array([0.80, 0.40, 1.50, 0.89, 1.50, 1.50, 1.50,
                    0.80, 0.40, 1.50, 0.89, 1.50, 1.50, 1.50])
# High-frequency amplitude, per joint rather than one scalar. The 5020 joints
# droop rather than draw current, so they need the amplitude raised to clear
# breakaway. The 4010 wrists have 5 N m against a ~6 Hz bandwidth, so they track
# the top segment properly and pull real torque: right_wrist_yaw tripped the
# torque guard at 4.6 N m on 0.25 rad.
HF_AMP = np.array([0.40, 0.40, 0.40, 0.40, 0.40, 0.16, 0.16,
                   0.40, 0.40, 0.40, 0.40, 0.40, 0.16, 0.16])

# rad. Droop is the mechanism, not a fault: it scales as tau/kp, and arm kp is
# 14.25 against the legs' 40-99, so one joint pulling 8 N m already droops
# 0.58 rad. Runaway is caught by MAX_VEL and TAU_GUARD_FRAC instead.
MAX_TRACK_ERR = 1.20
# Per-joint from the motor rating, not a flat number: 5020 is rated 37 rad/s and
# 4010 is 22, so a flat 8 rad/s would abort healthy excitation. Same mistake the
# flat 12 rad/s guard made on the legs.
VEL_LIMIT = np.array([37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,
                      37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0])
VEL_GUARD_FRAC = 0.50
TAU_GUARD_FRAC = 0.90  # fraction of spec torque
# Stick-slip release is a 15 ms torque spike and it is signal, not a fault: on
# right_wrist_pitch it hit 4.55 N m for 3 samples while the run mean was 1.02.
# Require the limit to be held before aborting; a real overload persists.
GUARD_HOLD = 10        # consecutive samples over limit before aborting

# Stepped frequencies, not a continuous chirp. Arm servos are softer than the
# legs relative to their load, so the top segment stays at 3 Hz: the intended
# 10 Hz bandwidth is really ~1.6 Hz once the real swung inertia is used, and
# the legs run already showed that excitation far above bandwidth buys nothing.
SEGMENTS = ((0.5, 12.0, None),    # damping, frictionloss: slow, large, reversals
            (1.0, 6.0, None),
            (2.0, 6.0, "hf"),     # armature: acceleration scales as f^2
            (3.0, 6.0, "hf"))


def smoothstep(n):
    """Min-jerk 0->1 over n samples: zero velocity and acceleration at both ends."""
    s = np.linspace(0.0, 1.0, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def segmented(lo, hi, dt, v_cap, hf_amp, amp_scale, taper=0.5, only=None,
              centre_override=None, amp_override=None):
    """Stepped-frequency excitation. Returns (n, 14) positions and a segment table."""
    centre = NEUTRAL.copy()
    amp_max = AMP_MAX.copy()
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
            amp = np.minimum(hf_amp, half)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--v-cap", type=float, default=3.0,
                    help="peak velocity cap for the low-frequency segments, rad/s")
    ap.add_argument("--hf-amp", type=float, default=None,
                    help="override HF_AMP for every joint, rad. Leave unset to "
                         "use the per-joint table, which derates the 4010 wrists.")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="fraction of config.Kp; Kd scales as sqrt to hold damping "
                         "ratio. Unlike the legs, leave this at 1.0: nominal arm kp "
                         "is 14.25, and halving it may not break stiction.")
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
                         "(e.g. left_elbow) or index 0-13.")
    ap.add_argument("--max-err", type=float, default=MAX_TRACK_ERR,
                    help="watchdog tracking-error limit, rad")
    ap.add_argument("--waist-kp-scale", type=float, default=1.0,
                    help="gain holding the waist at its current pose. Full gain "
                         "keeps the pelvis rigid with the harnessed torso, which "
                         "is what g1_arms.xml assumes. Do not lower this.")
    ap.add_argument("--iface", default="enp130s0")
    ap.add_argument("--out", default="arm_chirp_0.pt")
    ap.add_argument("--dir", default="/home/sid/projects25/src/pluto/sysid/data")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check the trajectory; never release or publish")
    args = ap.parse_args()

    if len(config.Kp) != N_ARM:
        raise SystemExit(f"config.Kp has {len(config.Kp)} entries, need {N_ARM}. "
                         f"Set robot = 'g1_arms' in config.py.")
    kp = config.Kp * args.kp_scale
    kd = config.Kd * np.sqrt(args.kp_scale)
    waist_kp = config.Kp[:3] * 0.0 + 28.50 * args.waist_kp_scale
    waist_kd = config.Kd[:3] * 0.0 + 1.814 * np.sqrt(args.waist_kp_scale)

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

    hf_amp = HF_AMP if args.hf_amp is None else np.full(N_ARM, args.hf_amp)
    q_ex, table = segmented(LIMIT_LO, LIMIT_HI, DT, args.v_cap, hf_amp,
                            args.amp_scale, only=only,
                            centre_override=args.centre, amp_override=args.amp)

    margin = 0.10
    ok = bool((q_ex > LIMIT_LO + margin).all() and (q_ex < LIMIT_HI - margin).all())
    print(f"trajectory {q_ex.shape}  {q_ex.shape[0] * DT:.1f}s  "
          f"within limits (>{margin} rad margin): {ok}")
    print(f"{'seg':>4}{'Hz':>6}{'s':>5}{'amp_elb':>10}{'v_pk':>7}{'a_pk':>8}{'revs':>6}")
    for i, (f, dur, amp, v, a, rev) in enumerate(table):
        print(f"{i:>4}{f:>6.1f}{dur:>5.0f}{amp[3]:>10.3f}{v[3]:>7.2f}{a[3]:>8.1f}{rev:>6d}")
    print(f"  total reversals per joint {sum(r for *_, r in table)}")
    print(f"  amplitude span {(q_ex.max(0) - q_ex.min(0)).round(3)}")
    print(f"  kp {kp.round(2)}")
    print(f"  kd {kd.round(3)}")
    print(f"  waist held at kp {waist_kp.round(1)} kd {waist_kd.round(2)}")
    if STICTION.any():
        stiction_err = STICTION / kp
        hf = hf_amp * args.amp_scale
        ratio = hf / np.maximum(stiction_err, 1e-9)
        print(f"  breakaway err (measured stiction / kp) {stiction_err.round(3)} rad")
        print(f"  hf amp / breakaway, per joint {ratio.round(1)}")
        thin = [NAMES[i] for i in np.where(ratio < 3.0)[0]]
        print(f"  under 3x breakaway: {thin if thin else 'none'}")
    else:
        print("  STICTION is all zeros: run stiction_arms.py and fill it in, "
              "otherwise the high-frequency segments may sit below breakaway")
    if not ok:
        raise SystemExit("trajectory exceeds joint limits -- lower --amp-scale or --hf-amp")

    ChannelFactoryInitialize(0, args.iface)
    state = {}
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: state.update(
            q=np.array([x.q for x in m.motor_state])[ARM_IDX],
            dq=np.array([x.dq for x in m.motor_state])[ARM_IDX],
            tau=np.array([x.tau_est for x in m.motor_state])[ARM_IDX],
            temp=np.array([x.temperature[0] for x in m.motor_state])[ARM_IDX],
            waist_q=np.array([x.q for x in m.motor_state])[WAIST_IDX],
            mode_machine=m.mode_machine), 20)
    t0 = time.time()
    while "q" not in state and time.time() - t0 < 5.0:
        time.sleep(0.02)
    if "q" not in state:
        raise SystemExit("no lowstate -- check --iface")
    q_start = state["q"].copy()
    waist_hold = state["waist_q"].copy()
    print(f"\ncurrent arm q {q_start.round(3)}")
    print(f"waist will be held at {waist_hold.round(3)}")
    print(f"mode_machine {state['mode_machine']}")

    if args.dry_run:
        print("\ndry run: nothing released, nothing published")
        return

    print("\nThis RELEASES the motion service and COMMANDS the arms.")
    print("The robot must be hoisted with both arms swinging clear.")
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
    tau_streak = np.zeros(N_ARM, dtype=int)
    vel_streak = np.zeros(N_ARM, dtype=int)
    next_t = time.time()
    for k, (q_des, gain) in enumerate(plan):
        # record state BEFORE the command: state k pairs with control k
        rec_q.append(state["q"].copy()); rec_dq.append(state["dq"].copy())
        rec_des.append(q_des.copy()); rec_tau.append(state["tau"].copy())
        rec_temp.append(state["temp"].copy())

        err = np.abs(state["q"] - q_des)
        tau_streak = np.where(np.abs(state["tau"]) > TAU_GUARD_FRAC * EFFORT,
                              tau_streak + 1, 0)
        vel_streak = np.where(np.abs(state["dq"]) > VEL_GUARD_FRAC * VEL_LIMIT,
                              vel_streak + 1, 0)
        over_tau = tau_streak >= GUARD_HOLD
        over_vel = vel_streak >= GUARD_HOLD
        if gain > 0.99 and (err.max() > args.max_err
                            or over_vel.any() or over_tau.any()):
            aborted = (f"watchdog at k={k}: max|err|={err.max():.3f} "
                       f"max|dq|={np.abs(state['dq']).max():.2f} "
                       f"vel over on {[NAMES[i] for i in np.where(over_vel)[0]]} "
                       f"tau over on {[NAMES[i] for i in np.where(over_tau)[0]]}")
            break

        for n, j in enumerate(ARM_IDX):
            mc = cmd.motor_cmd[j]
            mc.mode = 1
            mc.q = float(np.clip(q_des[n], LIMIT_LO[n] + 0.05, LIMIT_HI[n] - 0.05))
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(kp[n] * gain)
            mc.kd = float(kd[n] * gain)
        # waist at full gain throughout: this is what makes the pelvis and legs
        # rigid with the harnessed torso, which g1_arms.xml assumes.
        for n, j in enumerate(WAIST_IDX):
            mc = cmd.motor_cmd[j]
            mc.mode = 1
            mc.q = float(waist_hold[n]); mc.dq = 0.0; mc.tau = 0.0
            mc.kp = float(waist_kp[n]); mc.kd = float(waist_kd[n])
        for j in LEG_IDX:                       # legs hang limp
            mc = cmd.motor_cmd[j]
            mc.mode = 1; mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0
        for j in range(29, N_MOTOR):            # unused slots
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
        for j in range(N_MOTOR):
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
             full_q=q, full_des=des, kp=kp, kd=kd, waist_hold=waist_hold)

    e = np.abs(des[sl] - q[sl])
    print(f"\nwrote {out / args.out}  ({sl.stop - sl.start} samples at {HZ} Hz)")
    print(f"peak |dq|   {np.abs(dq[sl]).max(0).round(2)}")
    print(f"rom         {(q[sl].max(0) - q[sl].min(0)).round(3)}")
    print(f"track err   mean {e.mean():.4f}  max {e.max():.3f}")
    print(f"peak |tau|  {np.abs(tau[sl]).max(0).round(1)}  (spec {EFFORT})")
    print(f"temperature {temp[sl].mean(0).round(0)}")


if __name__ == "__main__":
    main()
