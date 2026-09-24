# PLUTO — what was actually done (for a LinkedIn post)

Timeline: 2026-09-17 to 2026-09-23. Unitree G1 humanoid, on loan, air-gapped.

---

## Draft post (short)

Spent the last week doing system identification on a Unitree G1 and then flying a
locomotion policy on it. It now walks, and I can drive it around with WASD.

The part I didn't expect: the biggest sim-to-real error wasn't friction or inertia.
It was that the robot only delivers 76–95% of the torque the model assumes it does.
Every identification pipeline I looked at treats the actuator gain as a known
constant and never fits it. I found it by regressing the robot's own reported
torque against the commanded PD torque — the one signal in the dataset that
doesn't come from the position trajectory the fit was already minimising.

Correcting it moved held-out prediction error more than armature, damping and
Coulomb friction combined.

Numbers: open-loop replay error on held-out data went 0.0365 → 0.0251 rad
(2.09° → 1.44°), against a measured run-to-run repeatability floor of 0.0051 rad.
It held up on a second holdout recorded at double the servo gains, which is how I
know the correction transfers across operating points rather than fitting one.

The other half was learning which parameters the data could determine at all.
Coulomb friction came out anywhere from 0.001 to 1.16 N·m across five fits with
under 3% change in predictive accuracy — unidentifiable. So I measured it
statically instead: hoist the robot, drop all torque, and record which gravity
torque each joint holds without moving. Armature disagreed 6–60x between left and
right legs that share an identical motor, which settles it as noise.

That identifiability analysis is what sized the domain randomisation for RL:
±5% on the parameter I could pin down, 0.2–2.0 N·m on the one I couldn't. sysID
centres the distribution, DR covers what's left. Randomising around a biased
centre doesn't help — a systematic 18% torque error points the same way every
sample, so the real robot sits outside the training distribution no matter how
wide you make the rest.

Policy: PPO in mjlab on the identified model, 32k iterations. Deployed over the
Unitree SDK at 50 Hz, observation is IMU + joint encoders only — no state
estimator, no contact sensors, no terrain scan.

Caveats, because they matter: flat ground, hoist strap slack as a catch, and it
deliberately stands still below ~0.4 m/s (the tracking reward makes standing worth
85% of max at 0.2 m/s — that's the reward function, not a bug).

---

## The full list, for reference

**System identification**
- Trimmed a 12-joint fixed-base G1 model; wired it into IIT-DLSLab's
  sim2real-robot-identification (`mujoco.sysid`)
- Designed the excitation: stepped-frequency blocks, per-joint amplitude caps set
  by a GRAVITY budget rather than joint limits (a soft servo pays C·sin(A)/Kp in
  droop, which binds before the limits do), phase offsets to decorrelate the
  regressor, self-collision verified over 2000 postures
- Wrote the hardware collection: unitree_hg over CycloneDDS, 200 Hz position
  control decimated from the robot's 1053 Hz state stream, ramped gain
  engagement, tracking/velocity watchdog, always-limp exit
- 7 recordings across 2 sessions; simulation-in-the-loop NLS over 36 parameters
  (Levenberg–Marquardt, FD Jacobians, Jacobian-scaled trust region)

**Results**
- Held-out open-loop RMSE 0.0365 → 0.0251 rad; second holdout at 2x gains 0.0246
- Repeatability floor 0.0051 rad, from a byte-identical repeat run
- Actuator gain shortfall 0.755–0.952 per joint; 30x cost reduction on a
  decoupled single-joint fit; two independent methods agreed within 6%
- Static breakaway friction measured directly: 0.16–0.85 N·m vs the model's 0.3

**Identifiability (the methodological part)**
- Left/right symmetry as a free standard-error estimate: damping agrees within
  14% (real), armature disagrees 6–60x (noise)
- Five fits under different settings to show friction spans three orders of
  magnitude at constant accuracy
- Rejected link-mass identification: it made held-out error WORSE than the nominal
  model. Cause: ∂τ/∂armature and ∂τ/∂link_inertia are the same Jacobian column,
  so adding mass made J'J singular
- Root-caused the gain error to a gain design that used rotor inertia instead of
  the inertia the joint accelerates (2–84x larger), making an intended 7 Hz
  ζ=2 servo actually 0.77 Hz ζ=0.22 — which is the 1 Hz resonance visible in the
  before/after plots. The same formula is in mjlab's upstream G1 config.

**Four hypotheses proposed and then falsified by measurement**
1. Unmodelled actuator delay — fitted in the interior, made the cost worse
2. A mechanically obstructed ankle — it was under-driven; 0.101 → 0.652 rad
3. Constant amplitude at high frequency buying a ∝ f² — useless at 4x the real
   servo bandwidth
4. Overestimated link inertia — rejected on holdout and on bound hits

**RL and deployment**
- Ported to mjlab as an override of the asset zoo — zero modifications upstream
- Per-joint actuator configs; action scale recomputed from the corrected stiffness
- 5 randomisation terms sized by the identifiability verdicts, plus command latency
- Dropped base_lin_vel from the actor so the policy needs no state estimator:
  its entire input is IMU + encoders
- 32k iterations PPO, 4096 envs, ~22 h
- Deployment node: ONNX + unitree_hg, no training-framework dependency, 50 Hz,
  per-joint watchdogs from the motor spec, always-limp exit
- Measured 60–80 ms command-to-response lag against the 0–20 ms trained for

**Hardware results**
- 60 s standing, zero aborts, torques 3–31% of limits
- Walks at 0.6 m/s; knee peaks 7.8 rad/s (46% of motor limit)
- Keyboard teleoperation: forward/back, turn, strafe, speed capped at 0.8 m/s

---

## Keep these caveats in whatever you post

- Flat ground only. No rough terrain policy was trained.
- Hoist strap slack as a catch throughout. Not a free-standing demo.
- Stands still below ~0.4 m/s by reward design, not by failure.
- The identification was done with the base welded and no ground contact, so it
  covers joint and actuator dynamics and says nothing about contact modelling.
- The retrain with the measured latency is queued, not done. The policy currently
  works DESPITE a 3-4x latency mismatch.
- One robot, one policy, one week. No statistics across seeds or units.
