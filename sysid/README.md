# G1 leg system identification

Measuring how the joints of a Unitree G1 humanoid actually behave, and closing the
gap to its MuJoCo model, so that a locomotion policy trained in simulation has a
chance of working on the hardware.

Twelve leg joints, robot hoisted with its feet clear of the floor, base treated as
welded. Seven recordings across two hardware sessions (2026-09-17 and 09-21), on a
robot on loan with no internet access.

## Result

Open-loop replay error — start the simulator from the robot's initial state, feed
it the same 30 s of commands, never correct it, and measure the gap:

| model | `chirp_0` | `chirp_kp10` | |
|---|---|---|---|
| MuJoCo nominal | 0.0365 rad (2.09°) | — | |
| identified | **0.0251 rad (1.44°)** | **0.0246 rad (1.41°)** | 1.46× better |

(`chirp_kp10` has no nominal-model baseline; it was recorded after the fit.)
| run-to-run repeatability | 0.0051 rad (0.29°) | | the floor no model can beat |

Both evaluation sets are **held out** — neither was used to fit. `chirp_0` differs
in amplitude; `chirp_kp10` runs at double the servo gains and double the torque, so
it tests a different operating point entirely.

The single largest improvement came from a parameter the fitting pipeline does not
fit: **the actuator gain**. The robot delivers only 82–95 % of the torque that
`Kp·(q_des − q) − Kd·q̇` implies. Correcting it moved the held-out error more than
armature, damping and friction combined.

## What was already known, and what this added

| parameter | before | after |
|---|---|---|
| `armature` | per-motor spec from rotor inertia × gear ratio | nothing usable — see identifiability below |
| `damping` | **0.0**, not modelled | hips 3.12 / 0.58 / 0.36 N·m·s·rad⁻¹ |
| `frictionloss` | flat 0.3 N·m guess | 0.16–0.85 N·m, **measured statically** |
| actuator gain | assumed exact | 0.755–0.952 per joint |

Final values, actuator order `L hip_pitch/roll/yaw, knee, ankle_pitch/roll` then `R`:

```python
frictionloss = [0.53, 0.59, 0.16, 0.67, 0.21, 0.21, 0.53, 0.59, 0.16, 0.85, 0.20, 0.21]
damping      = [3.123, 0.577, 0.359, 0.001, 1.457, 0.393, 3.032, 0.659, 0.330, 0.001, 0.808, 0.001]
gain_scale   = [0.952, 0.936, 0.860, 0.820, 0.755, 1.0, 0.943, 0.925, 0.847, 0.857, 1.0, 1.0]
# armature: use the motor spec (7520_14 0.0102, 7520_22 0.0251, 5020x2 0.0072),
# NOT the fitted values -- see below.
```

Machine-readable, with provenance and randomisation ranges:
`results/g1_legs_identified.json`.

## Identifiability — which numbers to believe

A fitted number is only a measurement if the data could actually determine it.
Three independent checks were used, and they disagree sharply by parameter.

**Left/right symmetry.** The two legs use identical motors, so mirrored joints must
share the same rotor inertia and friction. Damping on the six hip joints agrees
within 14 % across the mirror — a real measurement. Armature disagrees by 6–60×.
Two independent legs measuring the same physical quantity and landing 60 % apart
settles it: those numbers are noise.

**Repeated fits under different settings.** Coulomb friction came out 0.001, 0.115,
0.18, 1.16 and 0.53 across five fits — three orders of magnitude — with under 3 %
change in predictive accuracy. The trajectory genuinely cannot see it. So friction
was **measured instead of fitted**: hoist the robot, release all torque, and record
which gravity torque each joint holds without moving. That is a lower bound on
breakaway friction obtained with no estimator involved, and it is how we know the
knee's 0.001 N·m fit was wrong by a factor of ~850.

**Bound hits.** A parameter that ends its fit sitting on the edge of its allowed
range was not determined; the optimizer wanted to go further and something
physically impossible stopped it. Armature does this on five joints.

| parameter | verdict | randomise |
|---|---|---|
| actuator gain | identified; 30× cost change over ±20 %, two independent methods agree within 6 %, validated across a 2× gain change | ±5 % |
| damping, hips | identified; L/R within 14 % | ±30 % |
| damping, knees/ankles | not identified (pinned) | 0.01–2.0 |
| frictionloss | not identifiable from motion; statically measured | 0.2–2.0 N·m |
| armature | not identified (L/R 6–60×) | 0.5–3× spec |

The deliverable is therefore not one model but a **centred distribution with honest
error bars** — which is exactly what domain randomisation needs. Randomising around
a biased centre does not help: a systematic 18 % torque shortfall is in the same
direction on every sample, so the real robot sits outside the training distribution
no matter how wide the other ranges are.

## Method

1. **Excite.** Command a known trajectory and record what the joints actually did
   (`scripts/excite_legs.py`). Four fixed-frequency blocks — 0.5 and 1 Hz at large
   amplitude for velocity and direction reversals, 2 and 4 Hz at constant amplitude
   for acceleration. Armature is only visible through `q̈`, damping through `q̇`, and
   Coulomb friction only at velocity sign changes, so one frequency cannot excite
   all three.
2. **Fit.** Simulation-in-the-loop nonlinear least squares (`mujoco.sysid`): roll
   the simulator forward with candidate parameters, compare to the recording, adjust.
   The classical regressor `τ = Y(q, q̇, q̈)·θ` is unusable here — differentiating
   position twice at 200 Hz correlates the noise with the regressor and biases the
   estimate.
3. **Score on held-out data.** With 36 free parameters you can always make one
   recording look good. Identified link masses improved training slightly and made
   the holdout *worse than doing nothing*; they were rejected.
4. **Cross-check against physics.** The static friction measurement and the
   left/right symmetry test are independent of the optimizer, and both caught fits
   that looked fine by their own cost function.

The target is the **measured** motion, never the commanded motion. The real robot
lags its commands badly — soft servo, gravity droop, stiction — and the model is
supposed to lag the same way. A simulator that tracked commands perfectly would be
a simulator of a different robot.

## Reproduce

```bash
# regenerate the gain-corrected datasets from the raw recordings
python scripts/apply_gain_correction.py --use-stored

# the shipping fit: gain corrected, friction pinned at measured, armature+damping fitted
bash scripts/run_fit_final.sh

# score any dataset against the current model (--nominal resets it first, --show plots)
bash scripts/eval.sh chirp_0_gc
bash scripts/eval.sh chirp_kp10_gc --show
```

Needs `~/sysid_env` (`mujoco[sysid]`, MuJoCo ≥ 3.13 — `mujoco.sysid` does not exist
before that and `pip install mujoco` ships a stub whose ImportError is swallowed).
Hardware scripts need `~/g1_real_env` (`unitree_sdk2py` + CycloneDDS). Prefix
everything with `env -u PYTHONPATH` — a sourced ROS environment shadows numpy.

The fitting pipeline itself is [iit-DLSLab/sim2real-robot-identification](https://github.com/iit-DLSLab/sim2real-robot-identification),
expected at `~/projects25/src/sim2real-robot-identification` with `config.robot =
'g1_legs'`.

## Layout

```
data/       raw recordings, 200 Hz, seven keys each (time, dof_pos, dof_vel,
            des_dof_pos, des_dof_vel, kp, kd) plus a _meta.npz sidecar holding
            tau_est and per-joint temperature
  chirp_1     fitted        segmented excitation, Kp x0.5
  chirp_1b    fitted        byte-identical repeat -> the repeatability floor
  chirp_0     HELD OUT      smaller amplitude
  chirp_kp10  HELD OUT      Kp x1.0, ~2x torque
  knee_L                    single-joint, source of the gain discovery
  probe_0                   manual push; source of the static friction measurement
  drop_0                    passive release; the null result that started it
scripts/    excitation, recording, fitting, evaluation, gain estimation
results/    identified parameters, gain scale, synthetic-study answer key
FINDINGS.md full chronological log, including four falsified hypotheses
```

`*_gc.pt` files are gain-corrected derivatives, regenerated by
`apply_gain_correction.py --use-stored`, and are not kept in the repository.

## Safety, if you run the hardware scripts

The robot must be **hoisted with both feet clear of the floor**. Every parameter
here assumes a welded base and disabled collisions; a standing robot has ground
reaction forces the model does not contain, and the excitation would put it on the
floor.

Order matters: bring the robot low and engage damping *before* lifting. Lifting
while the AI balance controller is live makes it diverge — it sees no ground
reaction, concludes it is falling, and flails. `ReleaseMode()` zeroes all torque,
which is correct hanging and destructive standing.

`excite_legs.py` ramps gains in, ramps to the trajectory start, runs, ramps out, and
always leaves the joints limp including on abort. A watchdog stops it on excess
tracking error or velocity. `--dry-run` builds and limit-checks the trajectory
without releasing or publishing anything.

## Limitations

Every number here was measured on a leg hanging in the air: **no ground contact, no
body weight, no balancing**. This removes one class of sim-to-real error — joint and
actuator dynamics, the most common cause of locomotion transfer failure — and says
nothing about contact modelling. The next experiment worth doing is a contact-side
check, not more joint identification.

The remaining error is 4.8× the repeatability floor and is no longer in these
parameters. What is left is model *structure*: MuJoCo's single `frictionloss` term
cannot represent a harmonic drive whose breakaway friction exceeds its sliding
friction, and the ankles are a 4-bar parallel linkage modelled here as two serial
joints — which is why their per-motor `tau_est` does not map onto joint torque and
their gain scale had to be left at 1.0.

A synthetic validation study (`scripts/gen_synth.py`, `results/planted_truth.npz`)
plants known parameters and checks whether the estimator recovers them. It is
**unfinished**: the planted truth evaluates to a residual of 0.0125 instead of 0,
so generator and fitter still disagree on some convention. Until that closes, the
estimator has not been validated against a known answer — see FINDINGS.md #5.
