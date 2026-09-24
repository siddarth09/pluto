# PLUTO

Named for Sid's dog.

**System identification of a Unitree G1, and a locomotion policy trained on the
identified model and deployed to the real robot.** It walks at 0.6 m/s and drives
from the keyboard.

| | |
|---|---|
| held-out open-loop replay error | **0.0365 → 0.0251 rad** (2.09° → 1.44°) |
| second holdout, 2x servo gains | 0.0246 rad — generalises across operating points |
| run-to-run repeatability floor | 0.0051 rad |
| hardware | 60 s standing, walks at 0.6 m/s, keyboard teleop |

The largest sim-to-real error was the **actuator gain**: the robot delivers only
76–95% of the torque the model assumes, and the identification pipeline treats
that as a known constant it never fits. Correcting it moved held-out error more
than armature, damping and Coulomb friction combined. See
[`sysid/README.md`](sysid/README.md) for how, and [`sysid/MATH.md`](sysid/MATH.md)
for why.

---

## Layout

```
sysid/        system identification: data, scripts, findings, the maths
mjlab_g1/     RL training config and the hardware deployment node
assets/       Menagerie Go2 MJCF, from the earlier quadruped plan
RESUME_BRIEF.md, LINKEDIN.md   write-ups with accuracy guardrails
```

## Environments

Three, and they are not interchangeable. **Always prefix with `env -u PYTHONPATH`**
— a sourced ROS environment shadows numpy.

| env | for | why |
|---|---|---|
| `~/sysid_env` | fitting | `mujoco[sysid]` needs MuJoCo >= 3.13 |
| `~/bheema_rl_env` | mjlab training | torch + CUDA + mjlab |
| `~/g1_real_env` | anything touching the robot | `unitree_sdk2py` + CycloneDDS |

---

## 1. System identification

Needs the robot **hoisted, feet clear of the floor** — the model is fixed-base
with contacts disabled, so a standing robot invalidates it.

```bash
cd ~/projects25/src/pluto/sysid

# release the AI controller (robot must already be low and damped before hoisting)
env -u PYTHONPATH ~/g1_real_env/bin/python scripts/release_mode.py

# collect: stepped-frequency excitation, 30 s, writes data/<name>.pt
env -u PYTHONPATH ~/g1_real_env/bin/python scripts/excite_legs.py --out chirp_1.pt
env -u PYTHONPATH ~/g1_real_env/bin/python scripts/excite_legs.py --out chirp_1b.pt   # repeat -> noise floor

# estimate the per-joint actuator gain from tau_est, and write *_gc.pt copies
env -u PYTHONPATH ~/sysid_env/bin/python scripts/apply_gain_correction.py

# fit: gain corrected, friction pinned at the measured static values
bash scripts/run_fit_final.sh

# score against held-out data
bash scripts/eval.sh chirp_0_gc            # add --show for plots
bash scripts/eval.sh chirp_0_gc --nominal  # baseline: resets the model first
```

`--dry-run` on `excite_legs.py` builds and limit-checks the trajectory without
releasing or publishing anything. Use it whenever you change the excitation.

Identified values land in `sysid/results/g1_legs_identified.json`, with per-
parameter identifiability verdicts and the randomisation ranges they imply.

## 2. Train the policy

```bash
cd ~/projects25/src

# what is actually in the task you are about to train for hours
env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python -m pluto.mjlab_g1.preflight

env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python -m pluto.mjlab_g1.train \
  Pluto-Velocity-Flat-G1-Identified-DR --env.scene.num-envs 4096 \
  --video --video-interval 120000 --video-length 500

# watch a checkpoint (defaults to the newest run's newest checkpoint)
bash pluto/mjlab_g1/play.sh
bash pluto/mjlab_g1/play.sh 30000                                       # a specific iteration
bash pluto/mjlab_g1/play.sh 32000 Pluto-Velocity-Flat-G1-Identified-Delay80   # with measured lag
```

Tasks: `Pluto-Velocity-{Flat,Rough}-G1-Identified[-DR]`. The non-DR variants are
the debugging baseline — if training misbehaves, they separate "the model is
wrong" from "the randomisation is too wide". `-Delay80` injects the measured
hardware latency deterministically, for reproducing hardware behaviour in sim.

Knobs live in PLUTO, not in mjlab: `rewards.py` (weights, pose tolerances,
velocity curriculum), `randomisation.py` (DR ranges), `rl_cfg.py` (network,
iterations), `observations.py` (actor observation and history).

`--video-interval` is in environment steps: 1 iteration = 24 steps, so 5k
iterations = 120000.

## 3. Deploy to the robot

```bash
cd ~/projects25/src

# regenerate the deployment constants from the trained config
env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python -m pluto.mjlab_g1.export_deploy_map

# release the AI controller, or deploy.py will refuse to run
cd pluto/sysid && env -u PYTHONPATH ~/g1_real_env/bin/python scripts/release_mode.py && cd ..

# escalate in this order, each only after the previous looks right
cd ~/projects25/src
P="env -u PYTHONPATH ~/g1_real_env/bin/python pluto/mjlab_g1/deploy.py"
$P --dry-run                       # reads state, runs the policy, PUBLISHES NOTHING
$P --pose-only --kp-scale 0.5      # ramp to the default pose and hold it, no policy
$P --hold                          # policy closed-loop, zero velocity command
$P --vx 0.6 --duration 20          # walk
$P --teleop                        # keyboard driving
```

`deploy.py` is run as a **file, not with `-m`** — it deliberately has no mjlab
dependency (numpy + onnxruntime + `unitree_sdk2py` only), and `-m` would execute
the package `__init__` and import all of mjlab.

**Teleop keys**

```
w / s    forward / back        vx  +-0.8 m/s      space   stop
a / d    turn left / right     wz  +-0.7 rad/s    Ctrl-C  settle, then hold pose
<- / ->  strafe left / right   vy  +-0.4 m/s      Ctrl-C twice   release gains
```

Commands are hold-to-move and decay to zero 0.4 s after the last key, so walking
away stops the robot. The session exits after 60 s idle (`--idle-exit`).

It stands still below roughly 0.4 m/s **by design**: the tracking reward's
`std=0.5` makes standing worth 85% of maximum at a 0.2 m/s command, and the
`air_time` reward is gated at 0.5. To walk slowly, narrow `std` and lower that
gate in `rewards.py`.

---

## Safety

- **Hoisted** (feet clear) for system identification and for any first test of new
  code. **On the ground with the hoist strap slack as a catch** for policy runs.
- Bring the robot low and engage damping **before** lifting. Hoisting while the AI
  controller is live makes it diverge — it sees no ground reaction, concludes it is
  falling, and flails.
- `ReleaseMode()` zeroes all torque: correct hanging, destructive standing.
- Every commanded script ramps gains in, ramps to pose, and leaves the joints limp
  on abort or exception. Watchdogs are per joint, from the motor spec.
- A locomotion policy **hoisted is out of distribution** — unloaded legs accelerate
  far faster than trained. Good for validating plumbing, useless for judging
  behaviour.

## State and what is next

Working: identification, training, deployment, teleoperation.

Open:
- Retrain with the measured 60–80 ms command latency (`DELAY_MAX_LAG = 16`). The
  policy currently works *despite* a 3–4x mismatch.
- Low-speed walking, as above.
- Rough terrain: needs its own training run — the actor observation includes a
  187-point height scan, so it is a different input space and the flat checkpoint
  cannot be reused.
- **Contact is unvalidated.** Every identified parameter came from a leg hanging in
  the air: no ground contact, no body load. That is the largest untested part of
  the model.
- The synthetic estimator-validation study is unfinished (planted truth evaluates
  to 0.0125 instead of 0) — see `sysid/FINDINGS.md` #5.
