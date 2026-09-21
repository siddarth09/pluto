"""Release the G1 motion-control service so low-level commands are accepted.

Zeroes joint torque. Safe only with the feet clear of the floor -- on a standing
robot the legs go compliant and it collapses. Recover with `--restore`, which
reselects the `ai` controller.

    python scripts/release_mode.py            # show state, then prompt
    python scripts/release_mode.py --restore  # hand control back to `ai`
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/sid/unitree_sdk2_python")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except ImportError:
    from unitree_sdk2py.g1.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

IFACE = "enp130s0"
LEG_NAMES = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
             "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r"]


def sample(seconds: float = 2.0) -> dict:
    buf = []
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: buf.append((
            np.array([x.q for x in m.motor_state[:12]]),
            np.array([x.tau_est for x in m.motor_state[:12]]),
            np.array(m.imu_state.rpy),
        )), 50)
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.02)
    if not buf:
        raise RuntimeError("no lowstate messages -- check the interface and cable")
    return {"q": np.stack([b[0] for b in buf]),
            "tau": np.stack([b[1] for b in buf]),
            "rpy": np.stack([b[2] for b in buf])}


def report(s: dict) -> None:
    print(f"  imu rpy       {s['rpy'].mean(0).round(4)}")
    print(f"  sum |tau_est| {np.abs(s['tau'].mean(0)).sum():.2f} N.m")
    print(f"  q             {dict(zip(LEG_NAMES, s['q'].mean(0).round(3)))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--restore", action="store_true", help="reselect the `ai` controller")
    ap.add_argument("--iface", default=IFACE)
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    ms = MotionSwitcherClient()
    ms.SetTimeout(5.0)
    ms.Init()

    print("mode before:", ms.CheckMode())
    report(sample())

    if args.restore:
        print("\nSelectMode('ai') ...")
        print("  ->", ms.SelectMode("ai"))
        time.sleep(2.0)
        print("mode after:", ms.CheckMode())
        return

    print("\nReleaseMode() sets joint torque to ZERO.")
    print("Both feet must be clear of the floor and the space below must be empty.")
    print("The legs will drop and swing.")
    if input('Type RELEASE to proceed: ').strip() != "RELEASE":
        print("aborted")
        return

    print("  ->", ms.ReleaseMode())
    time.sleep(2.0)
    print("mode after:", ms.CheckMode())
    report(sample())


if __name__ == "__main__":
    main()
