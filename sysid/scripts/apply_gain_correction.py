"""Correct each joint's modelled actuator gain to the gain the robot actually delivers.

The pipeline treats Kp/Kd as known and never fits them, but regressing the robot's
own tau_est on the commanded Kp*(q_des-q) - Kd*qdot gives a slope well below 1 --
0.82 at the left knee, where an independent gain sweep put the optimum at 0.77.
An uncorrected gain error is a bias no joint parameter can absorb.

Writes `<name>_gc.pt` copies with kp/kd scaled per joint. Assumes both scale by
the same factor, which is what a single-slope regression can support.

Joints whose regression is not credible (low correlation, or slope outside
[MIN, MAX]) are left at 1.0 -- the parallel ankle linkage means per-motor tau_est
does not map onto the pitch/roll joint torques.
"""

import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path("/home/sid/projects25/src/pluto/sysid/data")
RESULTS = Path("/home/sid/projects25/src/pluto/sysid/results")
NAMES = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
         "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r"]
MIN_R, MIN_SLOPE, MAX_SLOPE = 0.90, 0.5, 1.2


def slopes(stem: str):
    d = torch.load(DATA / f"{stem}.pt", weights_only=False)
    meta = np.load(DATA / f"{stem}_meta.npz", allow_pickle=True)
    q, dq, des = (np.asarray(d[k]) for k in ("dof_pos", "dof_vel", "des_dof_pos"))
    kp, kd, tau = np.asarray(d["kp"]), np.asarray(d["kd"]), meta["tau_est"]
    pd_tau = kp[None, :] * (des - q) - kd[None, :] * dq
    s = np.array([np.polyfit(pd_tau[:, i], tau[:, i], 1)[0] for i in range(12)])
    r = np.array([np.corrcoef(pd_tau[:, i], tau[:, i])[0, 1] for i in range(12)])
    return s, r


def main() -> None:
    dry = "--dry-run" in sys.argv
    if "--use-stored" in sys.argv:
        z = np.load(RESULTS / "gain_scale.npz")
        scale = z["scale"]
        print("using stored gain_scale.npz:")
        for nm, sc in zip(NAMES, scale):
            print(f"  {nm:<9}{sc:>8.3f}")
        for q in sorted(DATA.glob("*.pt")):
            if q.stem.endswith("_gc") or "_g" in q.stem.replace("_gc", ""):
                continue
            d = torch.load(q, weights_only=False)
            if "kp" not in d:
                continue
            e = dict(d)
            e["kp"] = torch.as_tensor(np.asarray(d["kp"]) * scale, dtype=torch.float32)
            e["kd"] = torch.as_tensor(np.asarray(d["kd"]) * scale, dtype=torch.float32)
            out = q.with_name(f"{q.stem}_gc.pt")
            torch.save(e, out)
            print(f"  wrote {out.name}")
        return
    sources = [a for a in sys.argv[1:] if not a.startswith("-")] or ["chirp_1", "chirp_1b"]
    S, R = zip(*(slopes(s) for s in sources))
    slope, corr = np.mean(S, axis=0), np.mean(R, axis=0)

    credible = (corr >= MIN_R) & (slope >= MIN_SLOPE) & (slope <= MAX_SLOPE)
    scale = np.where(credible, slope, 1.0)

    prev = None
    store = RESULTS / "gain_scale.npz"
    if store.is_file():
        prev = np.load(store)["slope"]

    print(f"estimated from {sources}")
    hdr = f"{'joint':<9}{'slope':>8}{'r':>7}"
    hdr += f"{'prev':>8}{'ratio':>8}" if prev is not None else ""
    print(hdr + "  note")
    for i, nm in enumerate(NAMES):
        note = "" if credible[i] else "not credible (r or slope out of range)"
        row = f"{nm:<9}{slope[i]:>8.3f}{corr[i]:>7.3f}"
        if prev is not None:
            row += f"{prev[i]:>8.3f}{slope[i] / prev[i]:>8.2f}"
        print(row + "  " + note)
    if prev is not None:
        ok = corr > MIN_R
        print(f"\nmean |new/prev - 1| over credible joints: "
              f"{np.mean(np.abs(slope[ok] / prev[ok] - 1)) * 100:.1f}%")
        print("near 0% => the gain scale is a fixed fraction, portable to other Kp")
        print("large   => it is load-dependent and only valid at the Kp it was measured at")

    if dry:
        print("\ndry run: no _gc datasets written, gain_scale.npz untouched")
        return

    for p in sorted(DATA.glob("*.pt")):
        if p.stem.endswith("_gc") or "_g" in p.stem.replace("_gc", ""):
            continue
        d = torch.load(p, weights_only=False)
        if "kp" not in d:
            continue
        e = dict(d)
        e["kp"] = torch.as_tensor(np.asarray(d["kp"]) * scale, dtype=torch.float32)
        e["kd"] = torch.as_tensor(np.asarray(d["kd"]) * scale, dtype=torch.float32)
        out = p.with_name(f"{p.stem}_gc.pt")
        torch.save(e, out)
        print(f"  wrote {out.name}")
    np.savez(RESULTS / "gain_scale.npz", names=NAMES, scale=scale, slope=slope, corr=corr)


if __name__ == "__main__":
    main()
