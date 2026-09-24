# The math behind the G1 identification

Every result in `README.md` follows from the equations below. Numbers in the
worked examples are the measured ones.

---

## 1. What is being identified

Rigid-body dynamics with the base welded and no contact:

    M(q) q̈ + C(q, q̇) q̇ + g(q) = τ

The three fitted parameters enter per joint `j`:

    τ_j = [M(q) q̈]_j + [C q̇ + g]_j  +  a_j q̈_j  +  d_j q̇_j  +  f_j sign(q̇_j)
                                        ^armature   ^damping    ^Coulomb friction

Armature is not a separate force — it adds to the **diagonal of the mass matrix**:

    M_eff(q) = M(q) + diag(a)

That single fact explains why armature turned out unidentifiable. Its sensitivity
is `∂τ_j/∂a_j = q̈_j`, and relative to the link term `M_jj q̈_j` the effect is
`a_j / M_jj`. Measured on this robot:

    joint      M_jj      a       M_jj/a
    hip_pitch  0.8616  0.0102      84
    hip_roll   0.7034  0.0251      28
    knee       0.1150  0.0251       5
    ankle_p    0.0121  0.0072       2

At the hip pitch, armature is **1.2 % of the inertia the joint accelerates**. A
1.2 % effect cannot be resolved from a trajectory. At the ankle it is 50 %, and
indeed the ankle armature estimates were the sane ones.

## 2. The actuator, and the error we found

    τ_cmd = Kp (q_des − q) − Kd q̇          the model's assumption
    τ_real = α · τ_cmd,   α = 0.755…0.952   what the robot actually delivers

`α` is a **structural** error, not a parameter error: it multiplies the *input*,
while `a, d, f` multiply the *state*. No value of the fitted parameters can
absorb it — which is why the optimizer, given no `α` to turn, drove armature to
its lower bound and damping to its upper bound trying to fake the same effect.

## 3. Why the classical regressor fails here

Rigid-body dynamics is **linear in the inertial parameters**, so the textbook
method writes

    τ = Y(q, q̇, q̈) θ        →      θ̂ = (YᵀY)⁻¹ Yᵀ τ

one linear least-squares solve. It needs measured `τ` and measured `q̈`. We have
neither. `q̈` would come from differencing position twice, and at 200 Hz that
amplifies encoder noise by `1/dt² = 4×10⁴`.

The fatal part is not the variance, it is the **bias**. The noisy `q̈` appears
inside `Y`, not just in the residual — errors-in-variables. When the regressor
itself is noisy, least squares is *biased toward zero* and stays biased however
much data you collect. A biased estimator is worse than a noisy one.

## 4. Simulation-in-the-loop nonlinear least squares

Instead, roll the simulator forward and compare trajectories:

    θ̂ = argmin_θ  ½ Σ_k ‖ r_k(θ) ‖²,     r_k(θ) = ( y_sim,k(θ) − y_meas,k ) / s_k

`y_sim,k(θ)` is the solution of the ODE after `k` steps, so it depends on `θ`
nonlinearly — hence *nonlinear* least squares. `s_k = ‖y_meas‖₂/√2` per channel,
so a joint swinging 2 rad does not dominate one swinging 0.1 rad.

### Gauss–Newton and Levenberg–Marquardt

Linearise the residual, `r(θ + δ) ≈ r(θ) + J δ` with `J = ∂r/∂θ`. Minimising the
linearised squared norm gives the normal equations

    (JᵀJ) δ = −Jᵀ r                    Gauss–Newton
    (JᵀJ + μ diag(JᵀJ)) δ = −Jᵀ r      Levenberg–Marquardt

`μ → 0` is Gauss–Newton: long confident steps, valid only where the linearisation
holds. `μ → ∞` gives `δ → −Jᵀr/μ`, i.e. gradient descent with step `1/μ`. LM
interpolates, controlled by the **gain ratio**

    ρ = (actual reduction) / (reduction the quadratic model predicted)

`ρ ≈ 1` → trust the model, decrease `μ`. `ρ ≈ 0` → it over-promised, increase `μ`.
That is the `ratio` and `log10mu` columns in the fit log. `log10mu` climbing to 2
with steps of 1e-5 means the trust region collapsed: converged, or stuck.

`J` is built by finite differences — one extra rollout per parameter, 36 rollouts
per Jacobian. Hence 99.4 % of wall time in residual evaluation.

### Why `x_scale="jac"`

The parameters have incompatible units: armature ~1e-2 kg m², frictionloss ~1e0
N m, mass multipliers ~1e0, delay ~1e-3 s. A single trust-region radius in raw
units is simultaneously too large for one and too small for another. Scaling
columns of `J` makes the step isotropic in *effect* rather than in units.

## 5. Identifiability — the core of the whole project

Near the optimum the estimator covariance is

    Cov(θ̂) ≈ σ² (JᵀJ)⁻¹

so everything about "can this parameter be determined" is a statement about
`JᵀJ`. Two failure modes:

**(a) A near-zero column of J.** The parameter has no effect on the residual, its
variance is unbounded, and a bounded optimizer wanders until it hits a bound.
"Pinned at a bound" is the empirical signature of an unbounded variance.

**(b) Two nearly parallel columns.** Only a combination is determined; the
individual parameters are not. `JᵀJ` is near-singular in that direction.

The three sensitivity columns for one joint are

    ∂τ/∂a = q̈        ∂τ/∂d = q̇        ∂τ/∂f = sign(q̇)

and every identifiability result we got reads directly off them:

- **Coulomb friction needs velocity sign changes.** `sign(q̇)` is piecewise
  constant. While the joint moves one way it is a constant column — perfectly
  confounded with gravity and any other constant. Only *reversals* make it
  distinguishable. Hence designing the excitation for zero crossings.
- **Damping is confounded with the servo `Kd`.** The total torque contains
  `−Kd q̇` from the servo and `−d q̇` from the joint: *the same column*. With `Kd`
  free they are exactly unidentifiable. With `Kd` fixed the fraction recoverable
  is `d/(d + Kd)` ≈ 0.05/6.3 ≈ **0.8 %** on the weak joints.
- **Armature is confounded with link inertia.** `∂τ/∂a = q̈` and
  `∂τ/∂M_jj = q̈` — literally the same column, exactly collinear. This is why
  adding `--identify-link-mass` made the holdout *worse than the nominal model*:
  we added a parameter perfectly parallel to one already in the fit, making
  `JᵀJ` singular, and the optimizer put arbitrary mass into the null space.

### The left/right symmetry test

Mirrored joints share an identical motor, so `a_L = a_R` and `f_L = f_R`
physically. Two independent estimates of the same quantity, so their spread is an
empirical estimate of the estimator's standard error — no Jacobian required:

    damping   |d_L − d_R| / d̄  ≤ 14 %      → SE small, identified
    armature   a_L / a_R = 6…60×           → SE exceeds the value, noise

## 6. The servo, and the root cause of the gain error

One joint under position control:

    I q̈ + Kd q̇ + Kp q = Kp q_des
    ω_n = √(Kp / I)          ζ = Kd / (2√(Kp I))

Inverting gives the design rule that was used: `Kp = I ω_n²`, `Kd = 2 ζ I ω_n`,
with `ω_n = 2π·10` rad/s and `ζ = 2`.

**The error: `I` was taken as the armature, the rotor inertia alone.** The joint
actually accelerates `M_jj + a`. Substituting the true inertia, both quantities
scale by the *same* factor:

    ω_real / ω_design = √(a / I_eff)
    ζ_real / ζ_design = √(a / I_eff)

At the hip pitch `I_eff/a = 84`, so `√(1/84) = 0.109`:

    ω_real = 7.1 Hz × 0.109 = 0.77 Hz        (measured table: 0.77)
    ζ_real = 2.0   × 0.109 = 0.22            (measured table: 0.22)

One number per joint explains the whole table. An intended 7 Hz critically-
overdamped servo is in reality a **0.77 Hz servo with ζ = 0.22** — strongly
underdamped, which is precisely the 1 Hz hip-pitch resonance visible in the
before/after evaluation plots, and why the 4 Hz excitation segment was attenuated
to 13–60 %.

## 7. Bias versus variance, and why this decides the domain randomisation

Domain randomisation samples parameters from a distribution around a nominal.
That covers **variance** — uncertainty about where in the family the true system
sits. It cannot cover **bias** — the true system lying outside the family
altogether.

`α = 0.82` is bias. Randomising armature and friction ±50 % around a model that
over-delivers torque by 22 % leaves the real robot outside the training
distribution in every sample, because the error is in the same direction every
time. This is the whole argument for identifying before randomising:

    sysID   →  centres the distribution   (removes bias)
    DR      →  sizes the distribution     (covers variance)

and the identifiability analysis of §5 is what sets each width: ±5 % on `α` whose
Jacobian column is strong, 0.2–2.0 N·m on `f` whose column is nearly degenerate.
