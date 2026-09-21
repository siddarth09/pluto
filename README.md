# PLUTO

Named for Sid's dog.

Sim-to-real work on legged hardware.

- **[sysid/](sysid/)** — system identification of the Unitree G1's leg joints
  against its MuJoCo model. Identified parameters, held-out validation, and the
  randomisation ranges to carry into RL training. Start with `sysid/README.md`.
- **assets/** — Menagerie Go2 MJCF, from the earlier quadruped plan.

Next: an mjlab locomotion policy using the identified parameters as the nominal
model and the identifiability verdicts to size domain randomisation.
