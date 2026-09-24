# Skill distillation

Train specialists separately, then fit **one** network to all of them with a skill
selector appended to the observation. This is the only one of the multi-skill
options where you can add a skill *after* the fact without retraining the others.

```
collect.py        roll out a specialist, record (observation, action) pairs
train_student.py  fit one skill-conditioned network to every recorded skill
```

## Workflow

```bash
cd ~/projects25/src
PY="env -u PYTHONPATH PYTHONPATH=$PWD ~/bheema_rl_env/bin/python"

# one file per specialist
$PY -m pluto.distill.collect --task Pluto-Velocity-Flat-G1-Identified-DR \
    --checkpoint logs/rsl_rl/pluto_g1_velocity_identified/<run>/model_32000.pt \
    --skill 0 --skill-name walk --steps 200000

# ... repeat with --skill 1 --skill-name jump, etc.

$PY -m pluto.distill.train_student          # writes student.onnx + student_meta.json
```

Randomisation stays ON during collection: the student should see the same spread
of dynamics the specialist was trained to handle, not one nominal robot.

## Two things to be honest about

**Distillation only earns its keep when skills CONFLICT.** Flat and rough walking
do not — a single policy trained on both terrains is simpler and better. It pays
off for mutually exclusive behaviours: walk vs jump vs crouch. With one skill
recorded this pipeline is a plumbing test, not a capability.

**Behaviour cloning compounds error.** The student is fit only to states the
specialists visited. On its own rollout it drifts slightly, lands somewhere
unsupervised, drifts further. **A low validation loss does not mean it walks** —
only a rollout does. If it degrades, the fix is DAgger: roll out the STUDENT,
query the specialist for the action it would have taken in those states, add
those pairs, refit. Two or three rounds usually closes it.

## Deployment

`student.onnx` takes `[observation, skill one-hot]` and returns the same 29
actions, so `deploy.py` needs only a skill selector added to the observation
vector — the joint ordering, gains, watchdogs and teleop are unchanged.

Note this changes the input dimension, so `deploy.py`'s assertion that the ONNX
input matches the observation layout will need the skill width added.
