# PLUTO — system identification for the Unitree G1 legs

Named for Sid's dog.

Fitting joint-level dynamics parameters (rotor inertia, viscous damping, Coulomb
friction) of a MuJoCo G1 model so that simulation matches measured motion. The
point is the sim-to-real gap: an RL policy trained on a model whose actuators are
wrong learns to exploit dynamics the hardware does not have.

Scoped to the **12 leg joints** with the base welded. The torso and arm branch
share no dynamics with the legs once the base is fixed, so they are removed.

---

## 1. Layout

Two directories, deliberately separate:

    projects25/src/pluto/                      this project
      pluto/gen_synth.py                       synthetic dataset generator
      results/planted_truth.npz                the answer key
      README.md                                this file

    projects25/src/sim2real-robot-identification/   upstream fitting pipeline (IIT DLS)
      config.py                                robot selection + servo gains
      robot_model/g1_legs/                      trimmed 12-joint model + assets
      datasets/g1_legs/traj_0.pt                input recordings
      sysid_mujoco/my_fit.py                    the fit
      sysid_mujoco/eval_fit.py                  open-loop replay + metrics

`config.robot = 'g1_legs'` selects the model and the `Kp`/`Kd` arrays. Dataset
channel order follows actuator order: left hip_p, hip_r, hip_y, knee, ankle_p,
ankle_r, then the same for right.

### Environment

    ~/sysid_env          # has mujoco[sysid]; plain `pip install mujoco` ships a
                         # stub sysid module whose ImportError is swallowed

    cd ~/projects25/src/pluto
    env -u PYTHONPATH PYTHONPATH=$PWD ~/sysid_env/bin/python -m pluto.gen_synth

    cd ~/projects25/src/sim2real-robot-identification
    ~/sysid_env/bin/python -m sysid_mujoco.my_fit
    ~/sysid_env/bin/python -m sysid_mujoco.eval_fit

---

## 2. The math

### 2.1 What is being identified

Torque balance at a single joint, with the three fitted coefficients on the left:

    tau_applied  =  (I_link + armature) * qddot
                    + damping * qdot
                    + frictionloss * sign(qdot)
                    + gravity/coupling terms

- **armature** — reflected rotor inertia. The motor rotor's own inertia seen at
  the joint, scaled by the gear ratio *squared* -- which is why it is a large
  fraction of total joint inertia despite the rotor itself being light.
- **damping** — viscous friction, linear in velocity.
- **frictionloss** — Coulomb friction. Constant magnitude, sign follows the
  direction of travel.

The joints are driven by a position servo, not by commanded torque:

    tau_applied = Kp * (q_des - q) - Kd * qdot

`Kp`/`Kd` are treated as **known and fixed**, derived from the motor spec as
`Kp = armature_nominal * w^2`, `Kd = 2 * zeta * armature_nominal * w`, with
`w = 2*pi*10` rad/s and `zeta = 2`. See `config.py`.

### 2.2 Why not the classical regressor

Rigid-body dynamics is linear in the inertial parameters, so the textbook method
writes `tau = Y(q, qdot, qddot) * theta` and solves one linear least-squares
problem. That needs measured torque and measured acceleration. We have neither:
no joint torque sensors, and differentiating position twice at 200 Hz amplifies
noise into the regressor itself — which biases the estimate rather than just
adding variance, because the noise is correlated with the regressor.

### 2.3 Simulation-in-the-loop nonlinear least squares

Instead: roll the simulator forward with candidate parameters, compare the
resulting trajectory to the recording, and minimise the mismatch.

    theta_hat = argmin_theta  sum_k || r_k(theta) ||^2

    r_k(theta) = w * ( sim_k(theta) - measured_k ) / scale

Uses `mujoco.sysid`. Details as configured in `my_fit.py`:

- **Residual channels** — joint positions at weight 1.0, joint velocities at
  weight `--velocity-weight` (default **0.01**). Velocity enters the squared cost
  as `weight^2`, so the default contribution is 1e-4: this is effectively a
  position-only fit. Since damping's signal lives in velocity, raising this is
  the first lever to try if damping recovers badly.
- **Per-channel normalisation** — each signal is divided by
  `||measured||_2 / sqrt(2)`, so a joint that swings 2 rad does not dominate one
  that swings 0.1 rad. Channels whose norm is ~0 get unit scale instead; MuJoCo's
  default normalisation divides by zero there and silently drops the joint.
- **Free parameters** — 36 by default: armature, damping, frictionloss x 12
  joints. The link mass / CoM / inertia tensor / actuator delay families are
  gated behind `--identify-*` flags and off unless asked for.
- **Optimiser** — `sysid.optimize(..., x_scale="jac")`, backends `mujoco`,
  `scipy`, `scipy_parallel_fd`, 30 iterations by default. Levenberg–Marquardt /
  trust-region with finite-difference Jacobians: each column costs a full
  rollout, so cost is linear in the number of free parameters.
- **`x_scale="jac"`** — essential here. The free parameters have incompatible
  units (armature ~1e-2 kg m^2, frictionloss ~1e0 N m, link mass multipliers
  ~1e0, delay ~1e-3 s). A single trust-region radius in raw units is
  simultaneously too large for one and too small for another. Scaling by the
  Jacobian makes the step isotropic in *effect* rather than in units.
- **Bounds** — per-parameter, via `--armature-bounds`, `--damping-bounds`,
  `--frictionloss-bounds`. `move_off_bounds()` pushes the start 5% inside, since
  the optimiser flags anything within 0.1% of a bound. A parameter that comes
  back *sitting on* a bound is a failed fit, not a result.
- **Also fittable** — link mass scale, CoM offset, inertia tensor scale/shear,
  and one shared actuator delay. Adding these costs rollouts per iteration and
  introduces new confounds; keep them off until the three joint parameters are
  trusted.

### 2.4 Identifiability — the part that actually decides the outcome

Each parameter multiplies a different function of the motion:

| parameter | multiplier | needs |
|---|---|---|
| armature | `qddot` | high-frequency motion |
| damping | `qdot` | sustained speed |
| frictionloss | `sign(qdot)` | direction reversals |

Three consequences:

1. **A single frequency cannot excite all three.** Slow motion has no `qddot`;
   fast small motion has little net travel.
2. **Coulomb friction is only observable at reversals.** While direction is
   constant, `frictionloss * sign(qdot)` is just a constant offset,
   indistinguishable from any other constant. At a reversal it produces a
   discontinuity nothing else in the model can imitate.
3. **`damping` is structurally confounded with the servo `Kd`.** Joint damping
   contributes `-damping * qdot`; the servo contributes `-Kd * qdot`. Identical
   functional form. They are separable only because `Kd` is held fixed, and even
   then a planted `damping = 0.05` against `Kd = 6.3` is a 0.8% perturbation.
   **Expect damping to be the worst-recovered parameter class.** When it returns
   near its initial guess the honest conclusion is "not identifiable at this
   excitation level" — not "the fit is broken".

---

## 3. Excitation design

`excitation()` in `gen_synth.py`. Linear chirp, 0.2 Hz to 4 Hz over 20 s at 200 Hz.

- **Chirp, not a fixed sine** — the low end is velocity-dominated with slow zero
  crossings (damping, friction); the high end is acceleration-dominated
  (armature). One sweep covers both regimes.
- **4 Hz ceiling** — set by servo bandwidth (`w_n = 10 Hz`). Past a few Hz the
  plant attenuates the command and `qddot` stops growing, so the extra frequency
  buys nothing.
- **Phase is the integral of f(t)**, i.e. `2*pi*(f0*t + (f1-f0)*t^2/(2T))`.
  Writing `sin(2*pi*f(t)*t)` is the classic chirp bug — its true instantaneous
  frequency is `f0 + 2*(f1-f0)*t/T`, double the intended sweep at the end.
- **Amplitude falls as 1/f** to hold peak velocity near 3 rad/s, capped at 40% of
  each joint's half-range at the low-frequency end. See finding #2 below for why
  fixed amplitude fails.
- **Centred on each joint's range midpoint, not on q0** — the knee range is
  `[-0.087, 2.880]`, so a swing about zero clips on `ctrlrange` and flattens the
  velocity on the highest-inertia joint.
- **Per-joint phase offsets** (`linspace(0, 2*pi, 12)`) — joints moving in
  lockstep produce correlated regressor columns and a rank-deficient Jacobian;
  the fit then cannot attribute inertia to the right joint.
- **Self-collision is not a concern** — `build_fixed_base_model_xml` calls
  `_disable_all_collisions`, so a mid-range posture with large swings is safe.

---

## 4. The diagnostic table

Printed from a `traj_*.pt` before spending a fit on it. Left block: did the joint
move in the three ways the parameters need? Right block: is each parameter's
effect large enough to see?

| column | units | meaning |
|---|---|---|
| `pk\|dq\|` | rad/s | Peak velocity. **damping's multiplier** — near zero means damping is invisible. |
| `zc` | count | Velocity zero crossings. **frictionloss's multiplier** — the only places friction is observable. |
| `pk\|ddq\|` | rad/s^2 | Peak acceleration. **armature's multiplier.** |
| `errmax` | rad | `max\|q_des - q\|`. Not a multiplier — a sanity gauge. Large means the servo is saturated at `Kp*error` and dynamics are drowned out. Want < ~0.5. |
| `t_arm` | N m | planted armature x `pk\|ddq\|` |
| `t_damp` | N m | planted damping x `pk\|dq\|` |
| `t_fric` | N m | planted frictionloss (no multiplier — the magnitude *is* the parameter) |
| `t_srv` | N m | peak `\|Kp*(q_des-q) - Kd*qdot\|`, total servo torque |

**The ratio `t_* / t_srv` is what predicts recovery.** Same parameter, same
recording, opposite outcomes depending on how much torque the joint needed for
everything else: `L_hip_roll` damping is 0.22 against 25.2 N m (<1%, expect
failure) while `L_ankle_roll` friction is 0.87 against 1.7 N m (>50%, expect
clean recovery).

All entries are independent peak magnitudes at different instants. They do not
sum to a torque balance at any timestep — use them for orders of magnitude only.

`zc ~= 84` for every joint is a consistency check, not a coincidence: 0.2->4 Hz
over 20 s averages 2.1 Hz, giving ~42 cycles and two crossings per cycle.

---

## 5. Synthetic validation protocol

Before touching hardware data, validate the estimator against a known answer.

1. `planted_params()` picks ground truth: armature bases are the real reflected
   inertias per motor type (7520_14 = 0.0102, 7520_22 = 0.0251, 5020x2 = 0.0072)
   times a per-joint factor in [2.2, 4.0]; damping in [0.05, 0.60]; frictionloss
   `0.3 * [1.8, 3.5]`.
2. Roll out the excitation on that plant, save to `datasets/g1_legs/traj_0.pt`
   and the truth to `results/planted_truth.npz`.
3. Fit from the XML nominal (`armature 0.01, damping 0.0, frictionloss 0.3`).
4. Score recovered vs planted, per joint.

Two design rules that are easy to get wrong:

- **Nothing may be planted near the initial guess.** A planted `armature=0.0105`
  recovered as `0.0103` proves nothing — indistinguishable from a parameter the
  optimiser never touched. Hence the 2.2 multiplier floor: the smallest base
  (ankle, 0.0072) times 2.2 is the first value clearing 1.5x nominal for *every*
  joint.
- **Values must differ per joint**, for the same reason.

The generator calls the pipeline's own `build_fixed_base_model_xml` so generator
and fitter simulate a bit-identical plant. That makes this a test of the
*estimator*, with model mismatch deliberately set to zero. A good result here is
necessary, not sufficient — it says nothing about whether the model structure
matches hardware.

Note: the generated `<general>` actuators end up with **no `forcerange`** (the
rewrite copies `forcerange or ctrlrange` and the source used `inheritrange`), so
there is no torque saturation in this plant. Convenient — saturation is a
nonlinearity the fitter does not model — but do not read smooth tracking as
evidence the motion is physically reasonable for real hardware.

---

## 6. Findings log

**2026-09-17 — #1: `meshdir` is dropped when the fixed-base XML is relocated.**
`common.py:125` `_absolutize_file_attributes` rewrites every `file=` to
`source_dir / file` without consulting `<compiler meshdir>`, and an absolute
`file=` makes MuJoCo ignore `meshdir` entirely — so the `assets/` level vanishes
and meshes are looked up one directory too high. Breaks any Menagerie-style model
with a `meshdir`; go2 works only because it has none. Fixed on our side by
dropping `meshdir` from `g1_legs.xml` and inlining `assets/` into all 35 `file=`
attributes. Upstream untouched.

**2026-09-17 — #2: fixed-amplitude excitation is unusable on wide-range joints.**
35% of joint range put the command 1.9 rad from the actual position, so
`tau ~= Kp*error ~= 76 N m` against a planted frictionloss of 0.9 N m — a 1%
effect under a saturated actuator. Peak velocities of 21 rad/s were also
unphysical. Fixed by the 1/f amplitude law.

**2026-09-17 — #3: the startup step, which #2's fix did not address.**
The knee still hit 21.9 rad/s when the command asked for 3.0. Cause: the
trajectory is centred on the range midpoint (1.40 rad for the knee) but `MjData`
initialises `qpos = 0`, so every run opened with a 1.4 rad step into a Kp=99
servo and the ringdown dominated the record. Fixed in `rollout()` with
`data.qpos[:model.nu] = q_des[0]`. **`pk|ddq|` fell from ~1000 to ~40**: nearly
all the acceleration content I would have credited to the chirp was startup
transient. A diagnostic that looks healthy can be measuring an artefact.

**2026-09-17 — #4: removing `<sensor>` silently removes the objective function.**
`ValueError: need at least one array to concatenate` from
`processed_to_sysid_trajectory`. This pipeline defines its entire residual over
the *model's sensors*: `<jointpos>` sensors are the measurement channels, and
`add_joint_velocity_sensors` only mirrors joints that already have a position
sensor. With `nsensor = 0` there are zero residual columns. I had deleted the
block while trimming, reasoning it was tied to the old state layout like
`<keyframe>` — true of the base sensors, false of the joint ones. The builder
strips `accelerometer`/`gyro`/`framepos`/`framequat` but deliberately keeps
`jointpos`, which is why go2 worked. Fixed by adding 12 `<jointpos>` sensors
named `<joint>_pos`, matching go2's convention. Verified `nsensor` 0 -> 12, 24
after the velocity mirror, measurements `(4000, 24)`, controls `(4000, 12)`.

**2026-09-17 — #5: the planted truth is not the minimiser (open).**
The decisive check for a synthetic study: evaluate the residual at the planted
values. It must be ~0, because the dataset was generated by that exact plant.
It was not.

    fit start       y = 0.4487
    planted truth   y = 0.0479     <- should be 0
    fit result      y = 0.0105     <- 4.6x BETTER than truth

The optimiser fitting the data 4.6x better than the true parameters means it
spent its iterations bending 36 parameters to absorb a systematic replay
mismatch. Until `y(truth) ~= 0`, scoring recovered-vs-planted is meaningless:
the answer key is not what the objective rewards.

Note `y = 0.5 * sum(r^2)`, calibrated by reproducing the log's iter-0 value.

Cause found so far: `rollout()` recorded state k AFTER applying control k, while
the pipeline (`eval_fit.simulate_open_loop`) treats state k as BEFORE control k.
Fixed by recording before the step. That took `y(truth)` 0.0479 -> 0.0125, a 3.8x
improvement but not to zero.

A one-step shift of the old dataset only gave 0.0479 -> 0.0303, so the convention
was never the whole story. Remaining floor 0.0125. Leading suspect: `mujoco.sysid`'s
rollout interpolates the control TimeSeries between samples, whereas the generator
holds ctrl constant across the step (zero-order hold). Also unruled-out: the
residual's `predicted.resample(measured.times)`.

**Proposed fix:** stop hand-writing the rollout. Generate the synthetic dataset
through the pipeline's own rollout so generator and fitter are identical by
construction rather than by my matching their conventions one at a time.

**Why this matters for real data too:** on hardware there is no truth to check
against, so a systematic replay mismatch of this size would not announce itself
— it would just silently bias every identified parameter. The synthetic study
earning its keep is exactly this.

**2026-09-17 — #6: HARDWARE. The G1 legs are friction-dominated; the passive
drop test cannot work.**
First real-robot session. Robot hoisted, joystick in zero-moment, legs verified
free (manual push gave 2.62 rad of travel at L_hip_pitch, 2.69 at L_knee, peak
4.24 rad/s). Read-only throughout; nothing was ever commanded.

The finding: **released joints stop dead and stay where they are put.** L_knee
after the last manual release:

    q : 0.2368 0.2438 0.2412 0.2304 0.2231 0.2232 0.2230 0.2230 (rad, 0.2s apart)
    dq: 0.514 -0.002 -0.068 -0.097  0.006 -0.005 -0.005  0.006

No oscillation, no ringing, no decay envelope. Every joint ended wherever it was
last pushed (L_knee 0.3547 -> 0.2196), which also rules out a position controller
still holding setpoints.

Static friction lower bounds, from the gravity torque each joint holds without
moving (computed at the recorded hold pose, base pinned upright):

    R_knee   0.852      R_hip_p  0.527      R_hip_y  0.153
    L_knee   0.668      L_ank_p  0.202      L_hip_p  0.152
    R_hip_r  0.590      R_ank_p  0.198      (roll joints ~0: gravity-neutral,
                                             so their bound is vacuous)

Breakaway friction at the knees is >= 0.85 N m against the model's 0.3 nominal --
a measured ~3x sim-to-real gap, obtained with zero commands. It is consistent
with go2's published dynamic_friction (0.17-1.00) and with the planted range used
in the synthetic study (0.61-1.05), so that design choice validated.

**Consequence: the drop test is dead as designed.** Its premise was that a
displaced leg rings down under gravity, with exponential decay giving viscous
damping, linear decay giving Coulomb friction, and the oscillation frequency
giving inertia. There is no oscillation to decay -- Coulomb friction exceeds the
gravity torque, so the leg creeps and sticks instead of swinging. No frequency,
no envelope, no separation. What the passive data *does* give is the static
friction lower bounds above, and those are real.

Identification therefore requires **commanded** excitation: `ReleaseMode()` plus
low-level position control, where the servo supplies enough torque to break
stiction and sweep velocity through zero repeatedly.

Two follow-on consequences for the fit:
- `--frictionloss-bounds` default is (0.001, 2.0). If real values are near 2 the
  fit will sit on the upper bound. Widen to (0.001, 5.0).
- Re-plant the synthetic frictionloss higher (~1.0-2.5) to match what the
  hardware actually shows.

Also measured this session: lowstate publishes at **1053 Hz** (decimate by ~5 for
the pipeline's 200 Hz); `tau_est` **is** populated, which retracts the claim in
2.2 that no torque measurement exists -- the reasoning there was wrong even
though the conclusion (prefer sim-in-the-loop) stands, since `tau_est` comes from
motor current through a torque constant and still needs `qddot` by
differentiation. True sensor noise floor with nothing active: **9e-6 rad** on
position, **0.0074 rad/s** on velocity -- so noise is *not* a limiting factor
here and the "add measurement noise" step matters much less than first thought.
Joint temperatures 32-43 C, ankles warmest; Coulomb friction is
temperature-dependent, so log temperature with every dataset.

**2026-09-17 — #7: first commanded hardware dataset (`chirp_0.pt`).**
ReleaseMode succeeded (`CheckMode` name goes empty). 20 s chirp at half gains
(Kp 20/50/14, Kd scaled by sqrt(0.5) to preserve the damping ratio),
`--amp-scale 0.6 --v-peak 2.0`. Full run, watchdog never tripped.

**The actuator model is verified independently.** Regressing the robot's own
`tau_est` on the commanded `Kp(q_des-q) - Kd*qdot`:

    10 of 12 joints: r >= 0.98, slope 0.86-1.06, resid RMS 0.1-0.3 N m

So the commanded gains are the applied gains and the servo model is right. This
rules out a failure class: whatever the fit gets wrong is joint dynamics, not an
actuator mismatch it is silently absorbing.

Excitation: zc 106-533 (friction observable everywhere), pk|ddq| 29-69 rad/s^2
(comparable to synthetic 31-58), pk|dq| 0.63-1.65 rad/s, errmax 0.083-0.580.

Anomalies to chase:
- Right ankle `tau_est` is 0.56-0.70 of commanded PD torque (r 0.76-0.86) while
  the left ankle is 1.04-1.06 (r 0.98). Right ankle also moved MORE. Possibly
  the parallel ankle linkage / PR-mode mapping, where per-motor `tau_est` does
  not map 1:1 to pitch-roll joint torque.
- `L_ank_p` range of motion was **0.101 rad in two independent experiments**
  (manual push and commanded chirp) against 0.335 commanded. An identical number
  twice is mechanical obstruction, not dynamics. Inspect foot/harness.
- `L_hip_y` reached errmax 0.580 against the 0.60 watchdog: no margin.

**2026-09-17 — #8: first real fits FAIL, reproducibly. Delay is not the cause.**

    run                        final cost   bound hits (of 36)   termination
    no delay,  60 nfev            0.6697           16            max nfev
    +delay,   400 nfev            0.6905           19            ftol (converged)

Both start near 2.53-2.59, so only ~3.7x reduction (synthetic got 42x).

Pinning pattern: `armature` driven to its LOWER bound (0.001) on 7-8 joints while
`damping` is driven to its UPPER bound (2.0) on the SAME joints. `L_hip_roll`
maxes all three dissipative directions including frictionloss at the widened
5.0 ceiling.

**The independent cross-check rejects the result outright.** Measured `R_knee`
breakaway friction is >= 0.85 N m (finding #6, gravity only, no estimator). The
fit returns `frictionloss = 0.001` for that joint. Off by ~850x, toward "no
friction", for a joint we watched hold its own weight statically.

**Delay hypothesis: REJECTED.** I predicted the armature->0 / damping->max
signature meant an unmodelled phase lag (go2 needed `delay=1.349`). With
`--identify-delays` the delay landed in the interior -- so it was used -- and the
fit got *worse* (0.6905 vs 0.6697) with more pinning, while genuinely converging
on ftol at optimality 9e-5. Adding a real phase-lag DOF made it worse, so the
residual is not a timing problem. The signature is consistent with lag but not
diagnostic of it; I overstated it.

**Two localised defects remain unfixed and are the better suspects:**

1. `L_ank_p` is MECHANICALLY OBSTRUCTED -- exactly 0.101 rad of travel in two
   independent experiments (manual push, commanded chirp) against 0.335
   commanded. There is no model parameter for an obstruction, so the optimizer
   can only respond by maxing that joint's damping and inertia -- which is
   exactly the new pathology in the delay run (`L_ank_p` armature at the UPPER
   bound 0.4999, damping 1.999).
2. The G1 ankle is a PARALLEL MECHANISM (two actuators -> pitch+roll via a
   linkage; `mode_pr=0` means the controller does its own mapping), modelled
   here as two serial joints. Structural mismatch, localised to the ankles --
   and the ankles are exactly where the `tau_est`-vs-PD check failed (slope
   0.56/0.70 right ankle vs 0.96-1.06 elsewhere).

All 36 parameters are fit jointly through one coupled chain, so four bad ankle
joints can contaminate the eight good ones. Do NOT conclude "MuJoCo's friction
model cannot represent a harmonic drive" until these two are removed.

**Next:** (a) find and clear the left-ankle obstruction; (b) re-collect at
`--amp-scale 1.0 --v-peak 3.0` now that a run is known safe; (c) re-fit hips and
knees only; (d) treat the ankles as a separate parallel-linkage problem;
(e) override the initial guess at `common.py:751-756` -- frictionloss starts at
0.03 against a measured >= 0.85, i.e. 28x away in the parameter that dominates
this robot.

**2026-09-17 — #9: `chirp_1.pt` (segmented excitation). Two retractions and a
gain-design error.**

Full 30 s, no abort. Against `chirp_0`: pk|dq| 2.0-3.7 (was 0.6-1.7), pk|ddq|
111-549 (was 29-69), reversals 154-607 (was 106-533), errmax 0.41 (was 0.58).

**RETRACTED: `L_ank_p` was never mechanically obstructed (finding #8 item 1).**
Its rom is 0.652 rad here versus 0.101 before. It was under-driven, not blocked.
I built the obstruction claim on the number 0.101 appearing in two experiments
(a hand push and an under-amplitude command) and called matching decimals "a hard
stop, not a dynamics limit". It was a coincidence.

**RETRACTED: midpoint centring, not the watchdog, was the abort cause.** After
recentring on a hanging pose, errmax is 0.414 -- under even the original 0.60
threshold. The `--max-err 0.75` raise was unnecessary.

**The real constraint is amplitude vs the GRAVITY budget, not joint limits.**
Holding a joint at amplitude A costs ~C*sin(A), and a soft servo pays C*sin(A)/kp
in steady droop. Centring on range midpoints put hip_roll at 70 deg abduction:
~25 N m, i.e. 0.5 rad of droop at kp=49.6, which alone blew the watchdog.
Recentred on a hanging pose with per-joint `AMP_MAX` set by that budget, worst
predicted droop is 0.319 rad (hip_pitch, lowest kp of the big joints against the
longest lever). Measured errmax 0.414 -- prediction held.

**GAIN DESIGN ERROR: the closed-loop bandwidth is ~1 Hz, not 10 Hz.**
`Kp = armature * omega^2` with omega = 2*pi*10 uses the ROTOR inertia alone
(0.0102-0.0251). The link inertia seen at the joint is 30-50x larger (~0.4 kg m^2
about hip pitch), so the actual `omega_n = sqrt(kp/I) ~ 7 rad/s ~ 1.1 Hz`.

Consequence, visible per segment: the 4 Hz block is attenuated to 13-60% of its
commanded 0.26 rad span, and peak |ddq| is *highest in the 1 Hz block*
(152-549 rad/s^2) and LOW at 4 Hz (35-122) -- the reverse of the design intent.

So the redesign worked (3-8x more acceleration content) but **not by the
mechanism I claimed**. I predicted constant amplitude at high frequency would
give a ~ f^2; instead the gain came from the larger-amplitude 1 Hz block, because
4 Hz is four times past the real bandwidth. Next excitation should drop the 4 Hz
segment as wasted time and extend 1-1.5 Hz, or raise kp to buy bandwidth.

Also note: the largest |ddq| values (549 on L_hip_y, 398 on L_ank_r at 1 Hz) and
the largest reversal counts (353 on L_ank_r at 0.5 Hz) are probably stick-slip
chatter rather than smooth inertial content. Peak- and count-based metrics cannot
tell the two apart; that limitation already fooled me once (finding #8).

**2026-09-17 — #10: FIRST REAL RESULT. Open-loop RMSE halved on hardware data.**

`eval_fit` open-loop replay on `chirp_1.pt` (30 s, initialised once from sample 0,
never resynced), nominal model vs identified:

    mean RMSE  0.04797 -> 0.02323 rad   (2.07x)
    mean MAE   0.03455 -> 0.01788 rad

Per joint:

    joint            before    after  ratio
    L_hip_pitch      0.1052   0.0308   3.42
    R_hip_pitch      0.1060   0.0286   3.71
    L_ankle_pitch    0.0966   0.0315   3.07
    R_ankle_pitch    0.0583   0.0306   1.90
    R_hip_yaw        0.0314   0.0180   1.75
    R_ankle_roll     0.0288   0.0166   1.74
    L_hip_yaw        0.0333   0.0220   1.51
    R_hip_roll       0.0296   0.0204   1.45
    L_hip_roll       0.0261   0.0197   1.32
    L_ankle_roll     0.0257   0.0228   1.13
    L_knee           0.0182   0.0198   0.92   <- WORSE
    R_knee           0.0163   0.0180   0.91   <- WORSE

**Mechanism, visible in the plots:** the NOMINAL model *resonates* at 1 Hz on both
hip pitches -- simulated swings to +-0.55 rad while measured stays at +-0.2. The
fit kills it by driving hip_pitch damping to its 2.0 ceiling and frictionloss to
~1.1. Nearly the whole 2x came from removing that resonance, which is why
hip_pitch improves 3.4-3.7x and the well-modelled joints barely move.

**The knees got worse, and that explains their pinning.** Both knees had all
three parameters pinned at the LOWER bound, and they were already the
best-modelled joints (0.016-0.018 rad). A single global objective traded them
away for gains elsewhere: zeroing knee friction bought small improvements at the
hips. So the knee pinning is weak identifiability plus objective competition, not
evidence that knee friction is absent.

**Trajectory match does NOT imply correct parameters.** `R_knee` frictionloss
came out 0.001 against a directly measured >= 0.85 N m breakaway (finding #6),
while its trajectory matches well either way. Any per-joint parameter whose
trajectory is insensitive to it is unidentifiable regardless of how good the
overall fit looks. Physical cross-checks are the only defence.

**Actionable:** hip_pitch damping is PINNED at the 2.0 ceiling and is the single
parameter carrying most of the improvement -- it likely wants more. Widen
`--damping-bounds 0.001 10.0` and refit. Then evaluate on `chirp_0.pt` as a
HELD-OUT set: a fit that improves chirp_1 but not chirp_0 has learned the
excitation, not the robot.

**Plumbing added this session:** `my_fit` now writes `opt_params.yaml` /
`initial_params.yaml` (it previously saved nothing despite the `--output-dir`
help text); `robot_model/g1_legs/g1_legs_original.xml` is a frozen baseline for
`eval_fit --original`; `pluto/scripts/apply_fit.py` installs a saved fit into the
XML and flags pinned parameters and any frictionloss below measured stiction.

**2026-09-17 — #11: repeatability floor measured. 4.5x headroom remains.**
`chirp_1b.pt` is a byte-identical repeat of `chirp_1`'s command (verified:
`des_dof_pos` allclose). RMSE between the two MEASURED runs is the irreducible
floor -- no model can beat it.

    joint            repeat  nominal  fitted  fit/floor
    L_hip_pitch      0.0098   0.1052  0.0308        3.1
    R_hip_pitch      0.0121   0.1060  0.0286        2.4
    L_hip_roll       0.0051   0.0261  0.0197        3.8
    R_hip_roll       0.0058   0.0296  0.0204        3.5
    L_hip_yaw        0.0049   0.0333  0.0220        4.5
    R_hip_yaw        0.0050   0.0314  0.0180        3.6
    L_knee           0.0024   0.0182  0.0198        8.1
    R_knee           0.0029   0.0163  0.0180        6.3
    L_ankle_pitch    0.0044   0.0966  0.0315        7.1
    R_ankle_pitch    0.0036   0.0583  0.0306        8.4
    L_ankle_roll     0.0035   0.0257  0.0228        6.4
    R_ankle_roll     0.0017   0.0288  0.0166        9.6
    MEAN             0.0051   0.0480  0.0232        4.5

**The fitted model sits 4.5x above the noise floor**, so roughly 95% of the
remaining residual VARIANCE is model error rather than measurement noise. The
bottleneck is modelling, not data: no further hardware time is needed to make
progress.

**This corrects finding #10's reading of the knees.** I said they were "already
the best-modelled joints" and therefore weakly identifiable. Their ABSOLUTE error
is lowest (0.016-0.020), but relative to their own repeatability floor
(0.0024-0.0029) they are among the WORST at 6-8x. So the knees are badly
modelled, there is real unexplained signal there, and their three parameters
pinning at zero is a genuine failure rather than a well-fit joint with nothing
left to learn. Absolute RMSE without a per-joint noise floor is a misleading
metric -- comparing to the floor reverses the ranking.

Highest headroom (worst fit relative to floor): both ankle_rolls, ankle_pitches,
and knees. Lowest: the hip pitches, which is where the resonance fix already
delivered its 3.4-3.7x.

**2026-09-18 — #12: RESULT TABLE. Friction is unidentifiable; link mass is
rejected; four of my hypotheses are dead.**

Open-loop replay RMSE (rad). `chirp_1`+`chirp_1b` are the fit data; `chirp_0` is
never seen by any fit.

    model                                    train    HOLDOUT   free params
    nominal                                  0.0480    0.0365        --
    fit, all 36 free                         0.0226    0.0291        36
    fit, friction fixed at measured          0.0230    0.0299        24   <- CHOSEN
    + identified link masses                 0.0235    0.0404        36
    run-to-run repeatability floor           0.0051      --          --

**Link-mass identification REJECTED.** It made the holdout *worse than the
nominal model* (0.0404 vs 0.0365) with no training gain, and 6 of 12 masses ran
to their 0.5x/2.0x bounds -- not credible for CAD-derived masses. It was using
mass as a sponge for error elsewhere. This also kills my hypothesis that the
pinned hip/knee armature was caused by overestimated link inertia.

**Coulomb friction is not identifiable from this excitation.** Across three fits
`left_hip_pitch` frictionloss took 1.16, 0.18, and 0.53 (fixed) -- a 6x spread --
while predictive accuracy varied under 3%. Demonstrated, not argued.

**Constraining friction to the measured values costs ~nothing and is the model to
ship.** 12 fewer free parameters for +1.6% train / +2.7% holdout error, and the
values now agree with an independent static measurement. The free-friction fit is
nominally 2.7% better on the holdout, but we have no error bar on a single
holdout evaluation (the repeatability floor is 17% of the value), so the two are
equivalent within our precision. Prefer the physical one: it should transfer
better to velocities and loads outside the test, which is the entire point.

**Hypotheses I proposed and then refuted, in order:**
1. Unmodelled actuator delay (finding #8) -- delay fitted in the interior, cost
   got WORSE.
2. `L_ank_p` mechanically obstructed (#9) -- it was under-driven; rom 0.101 ->
   0.652 with more amplitude.
3. Constant amplitude at high f gives a ~ f^2 (#9) -- true in principle, useless
   here: servo bandwidth is ~1 Hz, so the 4 Hz block is attenuated to 13-60%.
4. Link inertia overestimated (this finding) -- rejected on holdout and bounds.

**What remains unexplained:** armature still pins at its lower bound on both hip
pitches and both knees, and the knees are the one place every fit does WORSE than
nominal (holdout 0.68-0.86x). Remaining candidates, untested: the inertia tensor
rather than mass (`--identify-inertia-tensor`); MuJoCo's single `frictionloss`
being unable to represent breakaway-vs-sliding friction in a harmonic drive; the
parallel ankle linkage modelled as serial joints.

**Tooling bug found and fixed:** `apply_fit.py` silently discarded all 12 mass
parameters ("NOT A JOINT PARAM") so the first link-mass evaluation tested
nothing. It now installs body masses and REFUSES to write a partial model. Its
first bound check for masses was also wrong -- compared the ratio against
absolute bounds, so it reported 0 pinned masses when 6 were pinned.

**2026-09-21 — #13: THE ACTUATOR GAIN WAS THE DOMINANT ERROR ALL ALONG.**

Per-joint excitation (`--only`, `--fit-joints`, plus residual masking) was built
to rescue the knees. It gave a clean NEGATIVE answer and then pointed at the real
problem.

Decoupled knee fit, 1.6 rad sweep, residual restricted to that joint: all three
knee parameters together move the objective **6%**. The knee's error is not joint
dynamics.

Then a sweep of the knee's ACTUATOR GAIN scale, everything else fitted:

    gain scale  final cost        gain scale  final cost
       60%      4.25e-04            82%       3.25e-05
       70%      6.01e-05            85%       6.28e-05
       75%      1.22e-05            90%       1.33e-04
       78%      1.07e-05  <- min   100%       3.21e-04  (as modelled)

**30x cost reduction, versus 6% from all three dynamics parameters.** Independent
confirmation: regressing the robot's own `tau_est` on the commanded
`Kp*(q_des-q) - Kd*qdot` gives slope 0.820 at that joint -- two unrelated methods
within 6% of each other.

Per-joint gain scale (mean of chirp_1, chirp_1b; dimensionless, measured/commanded):

    L_hip_p 0.952  L_hip_r 0.936  L_hip_y 0.860  L_knee 0.820  L_ank_p 0.755
    R_hip_p 0.943  R_hip_r 0.925  R_hip_y 0.847  R_knee 0.857
    ankle_roll x2 and R_ank_p: regression not credible (r 0.21-0.90, slope
    0.10-0.97) -- the parallel ankle linkage means per-motor tau_est does not map
    onto pitch/roll joint torque. Left at 1.0.

**Holdout result (chirp_0, never fitted):**

    nominal params, nominal gains        0.0365
    fitted params, nominal gains         0.0291
    fitted params, CORRECTED gains       0.0240   <- 1.52x vs nominal
    repeatability floor                  0.0051

**WHY the gain design was wrong.** `Kp = armature * omega^2` uses the ROTOR
inertia. The inertia the joint actually accelerates is 2-84x larger:

    joint     I_eff  armature   x   f_design  f_real  z_design  z_real
    L_hip_p  0.8616   0.0102   84    7.1Hz    0.77Hz    2.00     0.22
    L_hip_r  0.7034   0.0251   28    7.1Hz    1.34Hz    2.00     0.38
    L_hip_y  0.1284   0.0102   13    7.1Hz    1.99Hz    2.00     0.56
    L_knee   0.1150   0.0251    5    7.1Hz    3.30Hz    2.00     0.93
    L_ank_p  0.0121   0.0072    2    7.1Hz    5.47Hz    2.00     1.55

So instead of an overdamped 7 Hz servo everywhere we have an **underdamped
0.77 Hz servo at the hip pitch (zeta = 0.22)**. That is the quantitative
explanation of the 1 Hz hip-pitch resonance in finding #10's plots, and of why
the 4 Hz excitation segment was attenuated. (My earlier "bandwidth is ~1 Hz" was
right for hip_pitch only; it spans 0.77-5.5 Hz across joints.)

**Friction still does not recover even with the gain corrected.** Free-friction
fit on gain-corrected data returns L_hip_p 0.115 (measured 0.53), R_hip_p 0.001,
both knees 0.001. Coulomb friction is unidentifiable from position trajectories
here, full stop -- across five fits it has taken 0.001, 0.115, 0.18, 1.16 and
the measured 0.53 with under 3% spread in predictive accuracy. Ship it frozen at
the measured values.

**Tooling gap found:** `--fit-joints` froze the other joints' PARAMETERS but the
residual still summed over all 12 channels, so 11/12 of the objective was
un-improvable and swamped the 3 free ones (cost moved 0.18%). Freezing parameters
is not enough -- the objective has to be masked too. Fixed by zeroing those
sensors' weights; the same knee fit then moved 6%.

**2026-09-17 — go2 reference, for emulation.**
`datasets/go2/traj_0.pt` is real robot data with the same 7 keys, 200 Hz, 23.9 s.
Peak |dq| 2.5-5.2 rad/s and mean tracking error 0.091 rad -- our excitation sits
in the same regime (1.0-4.8 rad/s, 0.125 rad), so the design is comparable.

Their published result, `robot_model/go2/isaaclab_identified_parameters.py`:

    armature          0.0175 - 0.0385   (calf ~0.036 > hip/thigh ~0.021)
    viscous_friction  0.198  - 0.470    (= dof_damping)
    dynamic_friction  0.170  - 1.000    (= dof_frictionloss)
    delay             1.349
    Kp = 20.0   Kd = 1.5

Three things to copy: (1) they identified a delay, so `--identify-delays` is part
of the intended workflow, not an extra; (2) real viscous friction is 0.2-0.47,
i.e. far from the 0.0 XML nominal -- our planted damping range is realistic;
(3) their gains are much softer than the G1's RL gains (Kp 20 vs 40-99, Kd 1.5 vs
2.5-6.3). Softer servo = smaller `t_srv` = every joint parameter is a larger
fraction of the torque. Gains are ours to choose for a sysID experiment, and
lowering them is the single biggest identifiability lever available.

**2026-09-17 — my own broken diagnostic, recorded as a warning.**
First attempt at the truth-residual check built the model without
`actuator_gains`, giving `gainprm=1, biasprm=0` -- no servo, `ctrl` read as raw
torque -- and reported `sum r^2 = 131` at truth. `my_fit` builds the model twice
(lines 567 and 582), the second time with gains from the dataset's `kp`/`kd`.
Any harness that reproduces part of the fit must do the same.

**2026-09-17 — env: the fit requires `~/sysid_env`.**
`bheema_rl_env` and `g1_real_env` are both on mujoco 3.8.0; `mujoco.sysid` first
exists in 3.13 and needs the `[sysid]` extra. Not upgrading `bheema_rl_env` — its
regression gate asserts bit-identical torques and a MuJoCo bump would break it.

**2026-09-17 — non-bug, recorded to avoid re-investigating.**
`_rewrite_actuators_as_general` calls `_remove_all_by_tag(root, "motor")`, which
looks like it would leave the G1's `<position>` actuators in place and double
`nu` to 24. It does not: lines 173-174 then clear *every* child of `<actuator>`
regardless of tag. The `motor` call is redundant, not load-bearing. Verified
`nu=12 nq=12 nv=12`.

---

## 7. Model preparation

`robot_model/g1_legs/g1_legs.xml`, derived from the Menagerie G1:

- removed the `waist_yaw_link` body subtree (everything above the pelvis)
- removed the 17 non-leg actuators, leaving 12
- removed `<keyframe>` (tied to the old state layout)
- **kept** a `<sensor>` block of 12 `<jointpos>` sensors -- see finding #4; these
  are the fitting objective, not diagnostics
- inlined `meshdir` into `file=` paths (finding #1)

Verified: `nu=12 nq=19 nv=18 mass=18.18 kg` as a free-floating model;
`nu=12 nq=12 nv=12` after `build_fixed_base_model_xml` removes the freejoint.

`scene_flat.xml` exists because `get_robot_model_xml_path` expects a scene that
`<include>`s the robot, and follows the include to find the model.

---

## 8. Next

- [ ] Run `my_fit` on the synthetic set; score against `planted_truth.npz`.
      Pre-registered predictions: ankles recover best (`t_srv` only 1.4-2.7 N m);
      `L_hip_roll` and `R_hip_yaw` damping recover worst (<1% of servo torque);
      damping worst as a class.
- [ ] Repeat with measurement noise added to `dof_pos`/`dof_vel` — the synthetic
      set is currently noiseless, which flatters the estimator.
- [ ] Sensitivity/identifiability analysis from the converged Jacobian: condition
      number and the near-null-space directions, to confirm the damping/`Kd`
      confound is what the numbers say it is.
- [ ] Real G1 data collection over `unitree_hg` at 200 Hz (air-gapped robot; no
      internet on the robot network). Only after the synthetic fit is trusted.
- [ ] Refit on hardware data, then quantify the sim-to-real gap as open-loop
      replay RMSE before vs after (`eval_fit.py`).
- [ ] Feed the identified parameters into the mjlab training config.

## 9. Open questions

- Should `Kp`/`Kd` be fitted rather than fixed? They are nominal spec values, not
  measured. Fitting them would resolve the damping confound in one direction but
  adds 24 free parameters and makes `damping` and `Kd` jointly unidentifiable
  unless the excitation separates them.
- The base is welded. Real leg dynamics include base reaction; identified
  parameters may absorb a systematic error that only appears once the robot
  stands. Worth checking by refitting on a standing trajectory.
