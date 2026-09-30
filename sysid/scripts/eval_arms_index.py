"""Collect eval_arms plots and metrics into one comparison page."""

import json
import sys
from pathlib import Path

MODELS = {"N": "nominal everything",
          "A": "nominal armature/damping + measured friction",
          "B": "fitted armature/damping + measured friction"}
HELD_OUT = ("arm_shoulder_L", "arm_elbow_L")
FLOOR = 0.0163      # measured run-to-run repeatability, rad


def load(d: Path):
    f = d / "metrics.json"
    if not f.is_file():
        return None, {}
    m = json.loads(f.read_text()).get("metrics", {})
    return m.get("rmse_mean"), m.get("rmse_per_joint", {})


def main() -> None:
    out = Path(sys.argv[1])
    sets = sorted({p.name for m in MODELS for p in (out / m).glob("*") if p.is_dir()})
    table, perjoint = [], {}
    print(f"\n{'dataset':18s}" + "".join(f"{m:>10}" for m in MODELS) + "   split")
    for s in sets:
        vals = []
        for m in MODELS:
            v, pj = load(out / m / s)
            vals.append(v)
            perjoint[(m, s)] = pj
        split = "HELD OUT" if s in HELD_OUT else "in fit"
        cells = "".join(f"{v:10.5f}" if v is not None else f"{'-':>10}" for v in vals)
        print(f"{s:18s}{cells}   {split}")
        table.append((s, vals, split))

    joints = sorted(perjoint.get(("N", sets[0]), {}))
    if joints:
        print(f"\nper-joint RMSE on {sets[0]}:")
        print(f"  {'joint':28s}" + "".join(f"{m:>10}" for m in MODELS))
        for j in joints:
            row = "".join(f"{perjoint[(m, sets[0])].get(j, float('nan')):10.4f}"
                          for m in MODELS)
            print(f"  {j:28s}{row}")

    body = ["<h1>Arm sysid: three models</h1>",
            f"<p>Repeatability floor {FLOOR} rad; differences below it are noise.</p>",
            "<table border=1 cellpadding=6><tr><th>dataset</th>"
            + "".join(f"<th>{m}<br><small>{d}</small></th>" for m, d in MODELS.items())
            + "<th>split</th></tr>"]
    for s, vals, split in table:
        body.append(f"<tr><td>{s}</td>" + "".join(
            f"<td>{v:.5f}</td>" if v is not None else "<td>-</td>" for v in vals)
            + f"<td>{split}</td></tr>")
    body.append("</table>")
    if joints:
        body.append(f"<h2>Per-joint RMSE, {sets[0]}</h2><table border=1 cellpadding=6>"
                    "<tr><th>joint</th>"
                    + "".join(f"<th>{m}</th>" for m in MODELS) + "</tr>")
        for j in joints:
            body.append(f"<tr><td>{j}</td>" + "".join(
                f"<td>{perjoint[(m, sets[0])].get(j, float('nan')):.4f}</td>"
                for m in MODELS) + "</tr>")
        body.append("</table>")
    for s, _, split in table:
        body.append(f"<h2>{s} <small>({split})</small></h2>")
        for m in MODELS:
            p = out / m / s / "joint_trajectories.png"
            if p.is_file():
                body.append(f"<p><b>{m}</b> {MODELS[m]}<br>"
                            f"<img src='{p.relative_to(out)}' style='width:100%;max-width:1400px'></p>")
    idx = out / "index.html"
    idx.write_text("<html><body style='font-family:sans-serif'>" + "\n".join(body) + "</body></html>")
    print(f"\nplots + table: {idx}")


if __name__ == "__main__":
    main()
