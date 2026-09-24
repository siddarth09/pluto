"""Pre-flight check: confirm what the task you are about to train actually contains.

Run before committing to a long run. The failure this catches is training the
non-DR variant for hours by mistake -- the task ids differ by one suffix.

    python -m pluto.mjlab_g1.preflight
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pluto.mjlab_g1 as P  # noqa: E402  registers the tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

DR_TERMS = (
    "actuator_gain",
    "joint_armature",
    "joint_frictionloss",
    "joint_damping_hips",
    "joint_damping_knee_ankle",
)


def main() -> None:
    rl = load_rl_cfg(P.FLAT_DR_TASK)
    obs_dim = 96  # after dropping base_lin_vel; printed per-task below for the real value
    print("=== RL config (shared by all four tasks) ===")
    print(f"  actor hidden_dims   {tuple(rl.actor.hidden_dims)}")
    print(f"  critic hidden_dims  {tuple(rl.critic.hidden_dims)}")
    print(f"  max_iterations      {rl.max_iterations:,}")
    print(f"  num_steps_per_env   {rl.num_steps_per_env}"
          f"   -> 1 iter = {rl.num_steps_per_env} common steps")
    print(f"  save_interval       {rl.save_interval}"
          f"   -> {rl.max_iterations // rl.save_interval:,} checkpoints")
    print(f"  experiment_name     {rl.experiment_name}")
    print(f"  learning_rate       {rl.algorithm.learning_rate} ({rl.algorithm.schedule})")
    del obs_dim

    for task in (P.FLAT_TASK, P.FLAT_DR_TASK, P.ROUGH_TASK, P.ROUGH_DR_TASK):
        cfg = load_env_cfg(task)
        acts = cfg.scene.entities["robot"].articulation.actuators
        lag = max(a.delay_max_lag for a in acts)
        present = [t for t in DR_TERMS if t in cfg.events]
        dt = cfg.sim.mujoco.timestep
        print(f"\n{task}")
        print(f"  events            {len(cfg.events)}  ({len(present)}/{len(DR_TERMS)} DR terms)")
        print(f"  actuator groups   {len(acts)}")
        print(f"  max command lag   {lag} steps = {lag * dt * 1000:.0f} ms")
        print(f"  policy rate       {1 / (dt * cfg.decimation):.0f} Hz")
        actor = cfg.observations["actor"]
        hist = getattr(actor, "history_length", None)
        print(f"  actor obs         {list(actor.terms)}")
        print(f"  actor history     {hist if hist else 1} timestep(s)")
        stages = cfg.curriculum.get("command_vel")
        if stages is not None:
            for st in stages.params["velocity_stages"]:
                it = st["step"] // rl.num_steps_per_env
                rng = {k: v for k, v in st.items() if k != "step"}
                print(f"    curriculum @ step {st['step']:>9,} = iter {it:>7,}  {rng}")
        for name in present:
            t = cfg.events[name]
            rng = t.params.get("ranges") or (
                t.params.get("kp_range"),
                t.params.get("kd_range"),
            )
            print(f"    {name:<26} {t.mode:<8} {t.params.get('operation', 'scale'):<6} {rng}")
        if not present:
            print("    (no actuator/joint randomisation -- baseline variant)")


if __name__ == "__main__":
    main()
