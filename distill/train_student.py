"""Distil specialist policies into one skill-conditioned network.

Reads every data/skill*.npz from collect.py. The student sees the same observation
plus a one-hot skill selector, fit to the specialists' actions by regression.

    python -m pluto.distill.train_student

Behaviour cloning compounds error: the student drifts into states the specialists
never visited. Low validation loss does not prove it walks -- roll it out. The fix
is DAgger: roll out the student, query the specialist on those states, refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


class Student(nn.Module):
    def __init__(self, obs_dim: int, n_skills: int, act_dim: int, hidden):
        super().__init__()
        dims = [obs_dim + n_skills, *hidden, act_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.obs_dim, self.n_skills = obs_dim, n_skills

    def forward(self, x):
        # x = [obs, skill_onehot]; normalise only the observation part, so the
        # one-hot stays exactly 0/1 and the ONNX carries the statistics.
        obs, skill = x[:, : self.obs_dim], x[:, self.obs_dim:]
        obs = (obs - self.obs_mean) / self.obs_std
        return self.mlp(torch.cat([obs, skill], dim=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256, 128])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    files = sorted(DATA.glob("skill*.npz"))
    if not files:
        raise SystemExit(f"no skill*.npz in {DATA} -- run collect.py first")

    obs_l, act_l, sk_l, names = [], [], [], {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        k = int(d["skill"])
        names[k] = str(d["skill_name"])
        obs_l.append(d["obs"])
        act_l.append(d["action"])
        sk_l.append(np.full(len(d["obs"]), k, dtype=np.int64))
        print(f"  {f.name:<28} {d['obs'].shape[0]:>9,} pairs  skill {k} ({names[k]})")

    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)
    skill = np.concatenate(sk_l)
    n_skills = int(skill.max()) + 1
    obs_dim, act_dim = obs.shape[1], act.shape[1]
    print(f"\ntotal {len(obs):,} pairs   obs {obs_dim}  actions {act_dim}  "
          f"skills {n_skills}")
    if n_skills == 1:
        print("  NOTE: one skill only. This is a plumbing test -- distilling a "
              "single specialist is a no-op in capability terms.")

    dev = args.device if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(obs)
    Y = torch.as_tensor(act)
    S = torch.nn.functional.one_hot(torch.as_tensor(skill), n_skills).float()
    XS = torch.cat([X, S], dim=1)

    perm = torch.randperm(len(XS))
    n_val = int(len(XS) * args.val_frac)
    val, tr = perm[:n_val], perm[n_val:]

    model = Student(obs_dim, n_skills, act_dim, args.hidden).to(dev)
    model.obs_mean.copy_(X[tr].mean(0).to(dev))
    model.obs_std.copy_(X[tr].std(0).clamp_min(1e-6).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    XS, Y = XS.to(dev), Y.to(dev)
    for ep in range(args.epochs):
        model.train()
        idx = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            loss = nn.functional.mse_loss(model(XS[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(XS[val]), Y[val]).item() if n_val else float("nan")
        print(f"  epoch {ep + 1:>3}/{args.epochs}  train {tot / len(idx):.5f}  val {v:.5f}")

    # Per-skill error, so one skill cannot hide inside the average.
    model.eval()
    print("\nper-skill validation MSE:")
    with torch.no_grad():
        for k in range(n_skills):
            m = (torch.as_tensor(skill)[val] == k)
            if m.any():
                e = nn.functional.mse_loss(model(XS[val][m]), Y[val][m]).item()
                rms = e ** 0.5
                print(f"  skill {k} ({names.get(k, '?')}): MSE {e:.5f}  "
                      f"RMS action error {rms:.4f}")

    out = HERE / "student.onnx"
    torch.onnx.export(
        model.cpu(), torch.zeros(1, obs_dim + n_skills),
        str(out), input_names=["obs_skill"], output_names=["actions"],
        dynamo=False,
    )
    (HERE / "student_meta.json").write_text(json.dumps({
        "obs_dim": obs_dim, "n_skills": n_skills, "action_dim": act_dim,
        "skill_names": names, "hidden": args.hidden,
        "input_layout": "[observation (obs_dim), skill one-hot (n_skills)]",
    }, indent=2))
    print(f"\nwrote {out}  input {obs_dim + n_skills} -> {act_dim}")
    print("Validation loss does not prove it walks. Roll it out before believing it.")


if __name__ == "__main__":
    main()
