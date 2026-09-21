"""Write identified joint parameters from a fit into the robot XML.

`my_fit` reports results in an HTML page that eval_fit cannot read. This takes the
saved `opt_params.yaml`, writes explicit armature/damping/frictionloss attributes
onto each joint in a copy of the frozen baseline, and installs it as the model
eval_fit loads by default. The baseline stays available via `eval_fit --original`.

    python scripts/apply_fit.py <results_dir>/opt_params.yaml
    python scripts/apply_fit.py <...>/opt_params.yaml --dry-run
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SYSID = Path("/home/sid/projects25/src/sim2real-robot-identification")
sys.path.insert(0, str(SYSID))

from mujoco import sysid  # noqa: E402

ATTRS = ("armature", "damping", "frictionloss")
# Static friction lower bounds measured on hardware, README finding #6.
STICTION = {"left_hip_pitch": 0.53, "left_hip_roll": 0.59, "left_hip_yaw": 0.16,
            "left_knee": 0.67, "left_ankle_pitch": 0.21,
            "right_hip_pitch": 0.53, "right_hip_roll": 0.59, "right_hip_yaw": 0.16,
            "right_knee": 0.85, "right_ankle_pitch": 0.20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("params", type=Path, help="opt_params.yaml from a fit")
    ap.add_argument("--robot", default="g1_legs")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    base = SYSID / "robot_model" / args.robot / f"{args.robot}_original.xml"
    target = SYSID / "robot_model" / args.robot / f"{args.robot}.xml"
    if not base.is_file():
        raise SystemExit(f"missing frozen baseline {base}")

    pd = sysid.ParameterDict.load_from_disk(str(args.params))
    values = {name: float(np.atleast_1d(p.value)[0]) for name, p in pd.parameters.items()}
    bounds = {name: (float(np.atleast_1d(p.min_value)[0]),
                     float(np.atleast_1d(p.max_value)[0]))
              for name, p in pd.parameters.items()}

    tree = ET.parse(base)
    root = tree.getroot()
    joints = {j.get("name"): j for j in root.iter("joint") if j.get("name")}
    bodies = {b.get("name"): b for b in root.iter("body") if b.get("name")}

    rows, n_set, n_pinned = [], 0, 0
    for name, val in sorted(values.items()):
        lo, hi = bounds[name]
        # Mass bounds arrive absolute even though the CLI takes multipliers;
        # the ratio is reported for readability only.
        if name.endswith("_link_mass"):
            body_name = name[: -len("_link_mass")]
            body = bodies.get(body_name)
            inertial = body.find("inertial") if body is not None else None
            if inertial is None:
                rows.append((body_name, "mass", val, "", "NO <inertial> IN XML"))
                continue
            nominal = float(inertial.get("mass"))
            ratio = val / nominal
            # Bounds are stored absolute (kg), not as multipliers.
            at_bound = abs(val - lo) < 1e-4 * max(1.0, lo) or abs(val - hi) < 1e-4 * max(1.0, hi)
            n_pinned += at_bound
            note = f"{ratio:.2f}x nominal {nominal:g}" + (" PINNED" if at_bound else "")
            inertial.set("mass", f"{val:.6g}")
            n_set += 1
            rows.append((body_name, "mass", val, f"[{lo:g}, {hi:g}]", note))
            continue

        attr = name.split("_")[-1]
        if attr not in ATTRS:
            rows.append((name, attr, val, "", "UNHANDLED PARAMETER TYPE"))
            continue
        joint = name[: -(len(attr) + 1)]
        at_bound = abs(val - lo) < 1e-6 * max(1.0, abs(lo)) or abs(val - hi) < 1e-4 * max(1.0, abs(hi))
        n_pinned += at_bound
        note = "PINNED" if at_bound else ""
        if attr == "frictionloss":
            st = STICTION.get(joint.replace("_joint", ""))
            if st is not None and val < 0.5 * st:
                note = (note + " BELOW MEASURED STICTION %.2f" % st).strip()
        if joint in joints:
            joints[joint].set(attr, f"{val:.6g}")
            n_set += 1
        else:
            note = (note + " JOINT NOT IN XML").strip()
        rows.append((joint, attr, val, f"[{lo:g}, {hi:g}]", note))

    w = max(len(r[0]) for r in rows)
    print(f"{'joint':<{w}}  {'attr':<13}{'value':>10}  {'bounds':<16}note")
    for j, a, v, b, note in rows:
        print(f"{j:<{w}}  {a:<13}{v:>10.5g}  {b:<16}{note}")
    print(f"\n{n_set} attributes set, {n_pinned} of {len(values)} parameters at a bound")
    unhandled = [r for r in rows if "UNHANDLED" in r[4] or "NO <inertial>" in r[4]]
    if unhandled:
        raise SystemExit(f"{len(unhandled)} parameters could not be installed -- refusing to write a partial model")

    if args.dry_run:
        print("dry run: nothing written")
        return
    shutil.copy(target, target.with_suffix(".xml.prev"))
    tree.write(target, encoding="utf-8", xml_declaration=True)
    print(f"wrote {target}  (previous kept as {target.name}.prev)")
    print(f"baseline for comparison: eval_fit --original")


if __name__ == "__main__":
    main()
