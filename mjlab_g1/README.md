# G1 locomotion on the identified actuator model

mjlab velocity task using the system-identified G1 leg parameters from
[`../sysid/`](../sysid/README.md), with domain randomisation sized by how well
each parameter was actually determined.

Nothing here modifies mjlab. Upstream configs are imported and overridden, so
upstream fixes to the arms, keyframes, collision sets, rewards and terrain arrive
for free — only the leg actuator model differs.

## Tasks

| task id | terrain | randomisation |
|---|---|---|
| `Pluto-Velocity-Flat-G1-Identified` | plane | upstream only (6 event terms) |
| `Pluto-Velocity-Flat-G1-Identified-DR` | plane | + actuator/joint DR (11 terms) |
| `Pluto-Velocity-Rough-G1-Identified` | generated | upstream only |
| `Pluto-Velocity-Rough-G1-Identified-DR` | generated | + actuator/joint DR |

Non-DR variants exist as a debugging baseline — if training misbehaves, run the
non-DR flat task first to separate "the model is wrong" from "the randomisation
is too wide". Play mode never randomises; it would confound evaluation.

## Train

```bash
cd ~/projects25/src
env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python -m pluto.mjlab_g1.train \
  Pluto-Velocity-Flat-G1-Identified-DR --env.scene.num-envs 4096
```

Smoke test first with a small env count and a few iterations before committing to
a long run. Then play a checkpoint:

```bash
env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python -m pluto.mjlab_g1.play \
  Pluto-Velocity-Flat-G1-Identified-DR --checkpoint-file <path>
```

`train.py`/`play.py` are three-line shims: they import `pluto.mjlab_g1` (which
populates mjlab's task registry) and then call mjlab's own entrypoint. Using
mjlab's `train` console script directly will not find these tasks, because
nothing would have imported the registration.

## What differs from upstream's G1

`g1_identified.py` rebuilds the leg actuator configs. Per-joint rather than
per-motor-group, because `BuiltinPositionActuatorCfg.stiffness` is a scalar and
the measured gain scale varies within a group (`hip_pitch` 0.952 vs `hip_yaw`
0.860, both 7520_14).

| field | upstream | here |
|---|---|---|
| `stiffness` / `damping` | `armature * omega^2`, exact | **x gain_scale** (0.755–0.952) |
| `frictionloss` | unset (inherits XML) | measured static breakaway, 0.16–0.85 N·m |
| `viscous_damping` | unset (0) | fitted, hips 3.12 / 0.58 / 0.36 |
| `armature` | motor spec | motor spec, unchanged |
| action scale | `0.25 * effort / stiffness` | recomputed from the new stiffness |

Two of those deserve a note.

**The gain scale is the point.** The robot delivers 76–95 % of the torque that
`Kp*(q_des - q) - Kd*qdot` implies. Correcting it moved held-out replay error more
than armature, damping and friction combined. Uncorrected, the policy learns an
action-to-torque mapping the hardware does not have — a bias in the same direction
on every sample, which no randomisation of other parameters can cover.

**Action scale must be recomputed.** Upstream derives it from stiffness, so
reusing `G1_ACTION_SCALE` after lowering stiffness would silently shrink the torque
the policy can reach per unit action. `identified_action_scale()` recomputes it;
the knee goes 0.3507 -> 0.4276, exactly `1/0.820`.

**Armature is left at the motor spec on purpose.** Our fit for it is unusable —
mirrored left/right joints disagreed by 6–60x, which for joints sharing an
identical motor is conclusive noise. The spec is better information than a bad fit.

## Randomisation

`randomisation.py`. mjlab's velocity task already randomises foot friction,
encoder bias, base CoM, pushes and reset pose, but **nothing about the joints or
actuators** — so these five terms are additive, not replacements.

| term | function | range | why this width |
|---|---|---|---|
| `actuator_gain` | `dr.pd_gains` | x0.95–1.05 | identified: 30x cost change over ±20 %, two independent methods agree within 6 %, validated across a 2x gain change |
| `joint_damping_hips` | `dr.joint_damping` | x0.7–1.3 | identified: left/right agree within 14 % |
| `joint_damping_knee_ankle` | `dr.joint_damping` | 0.01–2.0 abs | not identified (pinned at bound) |
| `joint_frictionloss` | `dr.joint_friction` | 0.2–2.0 abs | not identifiable from motion: five fits spanned 0.001–1.16 with <3 % accuracy change |
| `joint_armature` | `dr.joint_armature` | x0.5–3.0 | not identified (6–60x left/right disagreement) |

Scoped to leg joints only. Nothing above the pelvis was identified, and applying
leg-derived ranges to the arms would be inventing data.

`mode="startup"` samples once per environment, matching mjlab's existing terms.
Switching `MODE` to `"reset"` resamples every episode and covers the ranges far
better for the same env count — the dr functions accept `env_ids`, so it is
supported. Worth trying; left at `startup` to match the pattern already known to
work here.

**Not randomised, and worth adding:** action latency. `ActuatorCfg` exposes
`delay_min_lag` / `delay_max_lag` / `delay_hold_prob`. We found no actuator delay
in the identification — fitting one made the residual *worse* — but that
experiment was a hanging leg over DDS at 200 Hz, and a deployed control loop has
latency the fixed-base setup never saw. It is a real sim-to-real gap we have no
measurement for, which is exactly the case for randomising it rather than
assuming zero.

## Limitation carried in from the identification

Every parameter here was measured on a **leg hanging in the air** — no ground
contact, no body weight, no balancing. This removes the joint-and-actuator class
of sim-to-real error, the most common cause of locomotion transfer failure, and
says nothing about contact modelling. Foot friction randomisation
(`foot_friction`, x0.3–1.2, upstream) is the only contact-side coverage, and it is
a guess rather than a measurement.
