"""Record G1 leg state into the dataset format sim2real-robot-identification expects.

Read-only: subscribes to rt/lowstate and publishes nothing. lowstate arrives at
~1 kHz and is decimated to `--hz` (the pipeline's config.frequency_collection).

For a passive (zero-torque) recording pass --passive: kp/kd are written as zeros
so the fitter simulates the same unactuated plant, and des_dof_pos is filled with
the measured angles since it is unused when the gains are zero.

    python scripts/record_lowstate.py --passive --duration 30 --out drop_0.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/sid/unitree_sdk2_python")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

N_LEG = 12
LEG_NAMES = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
             "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
             "right_hip_roll", "right_hip_yaw", "right_knee",
             "right_ankle_pitch", "right_ankle_roll"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=200.0, help="output rate after decimation")
    ap.add_argument("--iface", default="enp130s0")
    ap.add_argument("--passive", action="store_true", help="zero-torque recording: write kp=kd=0")
    ap.add_argument("--out", default="traj_0.pt")
    ap.add_argument("--dir", default="/home/sid/projects25/src/pluto/sysid/data")
    args = ap.parse_args()

    rows = []
    ChannelFactoryInitialize(0, args.iface)
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: rows.append((
            time.time(),
            np.array([x.q for x in m.motor_state[:N_LEG]]),
            np.array([x.dq for x in m.motor_state[:N_LEG]]),
            np.array([x.tau_est for x in m.motor_state[:N_LEG]]),
            np.array([x.temperature[0] for x in m.motor_state[:N_LEG]]),
        )), 200)

    print(f"recording {args.duration}s ...")
    t0 = time.time()
    while time.time() - t0 < args.duration:
        time.sleep(0.05)
    # Snapshot once: the DDS callback keeps appending, so deriving each column
    # from `rows` separately yields arrays of different lengths.
    snap = list(rows)
    if len(snap) < 10:
        raise SystemExit(f"only {len(snap)} messages -- check --iface")

    t = np.array([r[0] for r in snap]) - snap[0][0]
    q = np.stack([r[1] for r in snap])
    dq = np.stack([r[2] for r in snap])
    tau = np.stack([r[3] for r in snap])
    temp = np.stack([r[4] for r in snap])

    # Resample onto an exact uniform grid: the pipeline differences `time` for dt
    # and a jittered wall-clock stamp would bias every step.
    grid = np.arange(0.0, t[-1], 1.0 / args.hz)
    qs = np.stack([np.interp(grid, t, q[:, j]) for j in range(N_LEG)], axis=1)
    dqs = np.stack([np.interp(grid, t, dq[:, j]) for j in range(N_LEG)], axis=1)

    kp = np.zeros(N_LEG) if args.passive else None
    if kp is None:
        raise SystemExit("commanded recording not implemented yet -- use --passive")

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": torch.as_tensor(grid, dtype=torch.float64),
        "dof_pos": torch.as_tensor(qs, dtype=torch.float32),
        "dof_vel": torch.as_tensor(dqs, dtype=torch.float32),
        "des_dof_pos": torch.as_tensor(qs, dtype=torch.float32),
        "des_dof_vel": torch.zeros_like(torch.as_tensor(qs, dtype=torch.float32)),
        "kp": torch.as_tensor(np.zeros(N_LEG), dtype=torch.float32),
        "kd": torch.as_tensor(np.zeros(N_LEG), dtype=torch.float32),
    }, out / args.out)

    np.savez(out / (Path(args.out).stem + "_meta.npz"),
             joint_names=LEG_NAMES, tau_est=tau, temperature=temp,
             raw_time=t, raw_hz=len(snap) / t[-1])

    print(f"wrote {out / args.out}  ({len(grid)} samples at {args.hz} Hz, "
          f"raw {len(snap) / t[-1]:.0f} Hz)")
    print(f"peak |dq| {np.abs(dqs).max(0).round(2)}")
    print(f"range of motion (rad) {(qs.max(0) - qs.min(0)).round(3)}")
    print(f"temperature {temp.mean(0).round(0)}")


if __name__ == "__main__":
    main()
