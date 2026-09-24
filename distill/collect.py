"""Roll out a specialist policy and record (observation, action) pairs.

One file per skill. `train_student.py` then fits a single network to all of them
with a skill one-hot appended to the observation.

    python -m pluto.distill.collect --task Pluto-Velocity-Flat-G1-Identified-DR \
        --checkpoint <path> --skill 0 --skill-name walk --steps 200000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pluto.mjlab_g1  # noqa: F401,E402  registers the tasks
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--skill", type=int, required=True, help="skill index")
    ap.add_argument("--skill-name", required=True)
    ap.add_argument("--steps", type=int, default=200_000, help="total (env x time) pairs")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Randomisation ON during collection: the student should see the same spread
    # of dynamics the specialist was trained to handle, not one nominal robot.
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    agent_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    n_iter = max(1, args.steps // args.num_envs)
    O, A = [], []
    print(f"collecting {n_iter} steps x {args.num_envs} envs "
          f"= {n_iter * args.num_envs:,} pairs for skill {args.skill} ({args.skill_name})")
    with torch.inference_mode():
        for i in range(n_iter):
            try:
                action = policy(obs)
            except TypeError:
                action = policy(obs["actor"])
            O.append(obs["actor"].detach().cpu().numpy().astype(np.float32))
            A.append(action.detach().cpu().numpy().astype(np.float32))
            obs, _, _, _, _ = env.step(action)
            if i % max(1, n_iter // 10) == 0:
                print(f"  {i}/{n_iter}")

    O = np.concatenate(O)
    A = np.concatenate(A)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"skill{args.skill}_{args.skill_name}.npz"
    np.savez_compressed(path, obs=O, action=A, skill=args.skill,
                        skill_name=args.skill_name, task=args.task,
                        checkpoint=str(args.checkpoint))
    print(f"\nwrote {path}")
    print(f"  obs {O.shape}  action {A.shape}")
    print(f"  |action| mean {np.abs(A).mean():.3f}  max {np.abs(A).max():.3f}")


if __name__ == "__main__":
    main()
