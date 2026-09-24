# Resume brief — G1 system identification (PLUTO)

Factual summary for drafting resume/portfolio copy. Numbers are all verifiable in
`sysid/README.md`, `sysid/FINDINGS.md` and `sysid/results/g1_legs_identified.json`.

## One-line

System identification of a Unitree G1 humanoid's 12 leg joints against its MuJoCo
model, plus the domain-randomisation ranges derived from it, to reduce sim-to-real
gap for RL locomotion.

## Context and scope

- Hardware: one Unitree G1, on loan, air-gapped (no internet on the robot network).
- Two hardware sessions (2026-09-17, 2026-09-21), seven recordings.
- Robot **hoisted, feet clear of the floor**; fixed-base model (pelvis welded),
  collisions disabled. Legs only.
- Fitting pipeline: `mujoco.sysid` via IIT-DLSLab/sim2real-robot-identification.
- Training stack: mjlab (MuJoCo-Warp), PPO via rsl_rl.

## What was built

**Excitation design.** Four stepped-frequency blocks (0.5 / 1 / 2 / 4 Hz). Amplitude
capped per joint by a *gravity budget* rather than joint limits — holding a joint at
amplitude A costs ~C·sin(A) and a soft servo pays C·sin(A)/Kp in steady droop, which
is what actually binds. Per-joint phase offsets to decorrelate regressor columns;
segment tapering for continuity; verified self-collision-free across 2000 postures
with all collision geoms active.

**Hardware data collection.** Unitree SDK2 / CycloneDDS, `unitree_hg` messages.
Low-level position control at 200 Hz (decimated from the robot's 1053 Hz state
stream) with ramped gain engagement, min-jerk approach, a tracking-error and
velocity watchdog that always leaves joints limp on abort, and a guarded
motion-service release.

**Identification and validation.** Simulation-in-the-loop nonlinear least squares
(Levenberg–Marquardt, finite-difference Jacobians, Jacobian-scaled trust region)
over 36 joint parameters. Validated on two held-out datasets, one at a different
amplitude and one at **double the servo gains and ~2x the torque**.

**Port to RL training.** mjlab task configs using the identified actuator model,
with action scale recomputed from the corrected stiffness, and five added
randomisation terms. Implemented as an override of mjlab's asset zoo — zero
modifications to the upstream repository.

## Quantified results

| metric | value |
|---|---|
| Held-out open-loop replay RMSE | **0.0365 -> 0.0251 rad** (2.09° -> 1.44°), 1.46x |
| Second holdout, 2x servo gains | **0.0246 rad** — model generalises across operating points |
| Run-to-run repeatability floor | 0.0051 rad (0.29°), from a byte-identical repeat |
| Actuator gain shortfall found | 0.755–0.952 of commanded torque, per joint |
| Cost reduction from gain correction alone | **30x** on a decoupled single-joint fit |
| Static breakaway friction, measured | 0.16–0.85 N·m vs the model's flat 0.3 guess |

## The result worth leading with

The fitting pipeline treats the actuator gain as a known constant and never fits
it. Cross-checking the actuator model — regressing the robot's own reported torque
against the commanded `Kp·(q_des − q) − Kd·q̇` — showed it delivers only 76–95 % of
the assumed torque. Correcting that moved held-out error more than armature,
damping and Coulomb friction combined. Root cause: the gains were designed as
`Kp = armature·ω²` for `ω = 10 Hz, ζ = 2` using *rotor* inertia, while the inertia
the joint actually accelerates is 2–84x larger — so the real servo is 0.77 Hz,
ζ = 0.22 at the hip pitch, strongly underdamped. The same formula is in mjlab's
upstream G1 asset config.

## Identifiability analysis — the methodological contribution

Established which parameters the data could determine, using three checks
independent of the optimizer's own cost:

- **Left/right symmetry.** Mirrored joints share an identical motor. Damping agrees
  within 14 % across the mirror (a measurement); armature disagrees 6–60x (noise).
- **Repeated fits under varied settings.** Coulomb friction took 0.001, 0.115, 0.18,
  1.16 and 0.53 across five fits — three orders of magnitude — with under 3 % change
  in predictive accuracy. Unidentifiable. Replaced with a **static measurement**:
  hoist, release all torque, record which gravity torque each joint holds.
- **Held-out scoring.** Identified link masses improved training error slightly and
  made held-out error *worse than the nominal model*; rejected.

Outcome: 2 of 4 parameter classes are genuinely identified (actuator gain, hip
damping), 1 measured rather than fitted (friction), 1 left at the motor spec
(armature). Domain-randomisation widths are then set by that verdict rather than a
uniform guess — ±5 % on the identified gain, 0.2–2.0 N·m on unidentifiable friction.

## Hypotheses proposed and then falsified by measurement

Worth mentioning in interviews; it's the part that shows debugging rather than
pipeline-running.

1. Unmodelled actuator delay — fitted in the interior, made the cost *worse*.
2. A mechanically obstructed ankle — it was under-driven; range of motion went
   0.101 -> 0.652 rad with more amplitude.
3. Constant amplitude at high frequency buying acceleration ∝ f² — true in theory,
   useless at 4x the real servo bandwidth.
4. Overestimated link inertia — rejected on held-out data and on bound hits.

## Upstream/tooling fixes contributed

- `meshdir` dropped when the generated fixed-base XML is relocated (breaks any
  Menagerie-style model with a mesh directory).
- Fitted parameters were never persisted despite the CLI claiming to; added
  serialisation so evaluation can load them.
- Per-joint fitting: freezing parameters is insufficient because the residual still
  sums over all joints, so the frozen bulk swamps the free parameters. Added
  objective masking; the same fit then responded 30x more.

## In progress / next

- Train flat-terrain velocity policy in mjlab, with and without the derived
  randomisation, as an ablation.
- **Contact-side validation.** Everything above was measured on a leg in the air —
  no ground contact, no body load, no balancing. This is the largest untested part
  of the model and the next experiment worth running.
- Finish the synthetic validation study: planted ground-truth parameters currently
  evaluate to a residual of 0.0125 instead of 0, so the estimator has not yet been
  validated against a known answer.
- Hardware deployment of the trained policy.

## Do NOT claim (accuracy guardrails)

- **No policy has been deployed to hardware.** No sim-to-real transfer result exists
  yet. The 1.46x is *open-loop replay error of a hanging leg*, not transfer
  performance.
- **Do not say "identified 36 parameters."** Two of four parameter classes were
  shown to be unidentifiable; that finding is the stronger claim.
- **Do not claim contact or locomotion validation.** Fixed base, no contact.
- **Do not claim the synthetic study validated the estimator.** It is unfinished.
- **Do not call the excitation a "chirp."** It is stepped fixed-frequency; the swept
  version was replaced after the sweep's high-frequency end proved to be past the
  servo bandwidth. (Filenames still say `chirp`, which is a leftover.)
- Friction values are **lower bounds on breakaway** friction, not sliding friction.

## Keywords

MuJoCo · system identification · nonlinear least squares · Levenberg–Marquardt ·
identifiability analysis · domain randomisation · sim-to-real · Unitree G1 ·
Unitree SDK2 / CycloneDDS · low-level joint control · reinforcement learning (PPO) ·
mjlab / MuJoCo-Warp · held-out validation
