# PLUTO mimic — whole-body motion tracking on the identified G1

Tracking one clip (`J_Dance17_Shuffle`, 32 s) on the system-identified Unitree G1,
as an override of mjlab's tracking task. `~/mjlab` is not modified.

## Commands

```bash
cd ~/projects25/src/pluto/mimic

# 1. get a clip and resample it to 50 Hz. Once per clip.
./fetch_motion.sh --list                          # every clip in the Space
./fetch_motion.sh J_Dance17_Shuffle               # download + prep
./fetch_motion.sh B_SpinKarate --video            # also the mp4s
./prep_motion.sh /path/to/Already/Downloaded.npz  # prep only

# 2. train
./train.sh                                        # DR task, 4096 envs
./train.sh Pluto-Mimic-Dance17-G1                 # no-DR, no-delay baseline
./train.sh Pluto-Mimic-Dance17-G1-DR 2048         # fewer envs

# 3. watch it
./play.sh                                         # newest run, newest checkpoint
./play.sh 27000                                   # a specific iteration
./play.sh 29000 Pluto-Mimic-Dance17-G1            # a specific task

# 4. deploy (see deploy_mimic.py --help; read the safety notes first)
./deploy.sh --dry-run
./deploy.sh --probe-imu
./deploy.sh
```

Use the no-DR baseline when something looks wrong and you want to know whether
the randomisation is the cause.

Training logs land in `~/projects25/src/logs/rsl_rl/pluto_g1_mimic_dance17/`.
The runner exports an ONNX next to the checkpoints at the end of the run.

## Result, run 2026-09-29_10-22-11

29,000 iterations, 4096 envs, ~8 h. This run predates both fixes in the section
below, so it has no arm delay and was graded on the unlearnable root-position term.

| Metric | Value | Reads as |
|---|---|---|
| `error_body_pos` | 0.0491 m | 4.9 cm mean over 14 tracked bodies |
| `error_body_rot` | 0.1376 rad | 7.9 deg |
| `error_anchor_rot` | 0.0860 rad | 4.9 deg torso orientation |
| `error_anchor_pos` | 0.4852 m | global drift, and it is expected -- see below |
| episode length | 488.6 / 500 | completes the clip |
| `anchor_pos` / `anchor_ori` terminations | 0.0000 | never loses the reference |

Global drift is not a failure: with `has_state_estimation=False` the policy cannot
observe where it is, so it tracks the *shape* of the dance and wanders. See
`~/Desktop/pluto-mimic-rewards.html` for the full reward walkthrough.

## Getting other clips

`./fetch_motion.sh --list` prints all 60 clips in the
[`exptech/g1-moves`](https://huggingface.co/spaces/exptech/g1-moves) Space --
`J_Dance*` and `B_*Dance*` for dancing, `B_*Karate*`, `B_Fence*`, plus a few
others. `./fetch_motion.sh <ClipName>` downloads just that clip's npz and
resamples it in one step.

Each clip in the Space has four artefacts; only the first is needed for training:

| path | what |
|---|---|
| `training/<Clip>.npz` | the retargeted motion, 60 fps |
| `capture/<Clip>.mp4` | the source human video |
| `retarget/<Clip>_retarget.mp4` | the retarget preview |
| `policy/<Clip>_policy.onnx` | someone else's trained policy for it |

To train on a different clip, point `DEFAULT_MOTION` in `env_cfgs.py` at the new
`motions/<Clip>_50hz.npz` and rename the tasks in `__init__.py`.

The npz needs no retargeting: 29 joints in the G1's SDK order, 30 bodies
`pelvis .. right_wrist_yaw_link`, MuJoCo `wxyz` quaternions. Verified by setting
the model's qpos from the file and comparing forward kinematics against the
stored body positions -- agreement under a micron.

## Tasks

| Task | Randomisation | Delay | `base_lin_vel` in actor | Use |
|---|---|---|---|---|
| `Pluto-Mimic-Dance17-G1` | no | no | no | debugging baseline |
| `Pluto-Mimic-Dance17-G1-DR` | yes | 80 ms | no | **train this** |
| `Pluto-Mimic-Dance17-G1-StateEst` | yes | 80 ms | yes | upper bound, not deployable |

## Layout

```
review/              15 papers, read in the order below
motions/             the 50 Hz clip the loader actually consumes
prep_motion.py       60 -> 50 Hz resampler, with FK regeneration
env_cfgs.py          identified robot + motion file + DR, over mjlab's tracking cfg
randomisation.py     arm and waist terms, widths from the arm identification
rl_cfg.py            PPO, (1024,512,256,128)
__init__.py          task registration
train.py / play.py   entrypoints that import this package before mjlab's CLI
```

## Three things that were not obvious

**The clip had to be resampled.** mjlab's `MotionLoader` never reads the `fps`
key — it consumes one frame per policy step. The g1-moves clips are 60 fps and
this env runs at 50 Hz, so playing one as-is stretches the motion by 1.2x and
leaves the stored velocities inconsistent with the positions being tracked.
`prep_motion.py` follows mjlab's own `csv_to_npz` convention: lerp positions,
slerp orientations, central-difference the velocities. A cubic-spline derivative
is more "correct" and inflates peak joint velocity 16.8 -> 26.5 rad/s, because
the source velocities are themselves smoothed central differences.

**The clip is already retargeted, and verified.** `J_Dance17_Shuffle.npz` is not
raw mocap. Its 29-joint axis is the G1's SDK joint order and its 30-body axis is
`pelvis .. right_wrist_yaw_link`, confirmed by setting the model's qpos from the
file and comparing forward kinematics against the stored body positions:
agreement to under a micron. There is no retargeting step in this pipeline.

**`has_state_estimation=False` is the default here.** The hardware has no base
linear velocity estimate, so `base_lin_vel` and `motion_anchor_pos_b` are dropped
from the actor. That is the same choice PLUTO's velocity policy made and the
reason it was deployable. The `StateEst` task keeps them as an upper bound on how
much that costs.

## Randomisation widths, and where they come from

Legs come from `pluto/mjlab_g1/randomisation.py` unchanged. Arms and waist are
added here, sized by the 2026-09-28 arm identification:

| Parameter | Range | Basis |
|---|---|---|
| arm gain | 0.92–1.06 | **measured**: 12 of 14 joints, two runs agreeing to 0.002 |
| arm frictionloss | 0.15–0.80 N·m | **measured**: 6 of 7 mirrored pairs agree within 21% |
| shoulder_roll friction | 0.40–2.50 N·m | **unresolved**: 2.19 left, over the 3.75 ceiling right |
| arm armature | 0.5–3.0× | **not identified**: 6 values pinned at bounds, mirrors disagree 2–150× |
| waist (all) | wide | **never identified** — it was clamped at full gain to make the arm experiment valid |

Unlike the legs, the arms needed no gain *correction* — they already deliver
0.92–1.06 of commanded torque, against the legs' 0.755–0.952. So the model keeps
nominal arm gain and the range above is uncertainty about it, not a re-guess.

## Reading order

1. `01_deepmimic` — the ancestor. The tracking reward decomposition, reference
   state initialisation, early termination, phase. Every term in mjlab's
   `tracking/mdp/rewards.py` traces back here.
2. `02_amp` — the contrast: a style discriminator instead of frame-by-frame
   tracking. Read it to understand what tracking gives up and gains.
3. `04_beyondmimic` — **this stack**. G1, LAFAN1, built on mjlab, full sim-to-real.
   Its finding that Cartesian body-position encoding beats joint-angle encoding
   is why the motion file carries `body_pos_w` and why the config names 14
   specific bodies and an anchor.
4. `14_mjlab` — the framework itself.
5. `05_what_matters_gmt` — ablations, once you have questions about which knobs matter.
6. `07_exbody` / `08_exbody2` — asymmetric tracking: loose upper body, strict legs.
   Directly relevant given the legs are well identified and the arms only lightly.
7. `09_h2o` / `10_omnih2o` / `11_humanplus` — retargeting and teleop pipelines end to end.
8. `12_phc` / `13_sonic` / `15_limmt` — scaling from one clip to many.
9. `06_phuma` — retargeted is not the same as physically feasible. Read before
   trusting a large motion dataset.

Then read `~/mjlab/src/mjlab/tasks/tracking/mdp/` — `rewards.py`, `commands.py`,
`terminations.py`. With DeepMimic behind you, every term will be recognisable.

## Deployment

The exported ONNX is self-contained. Inputs are `obs[154]` and `time_step`;
outputs are `actions[29]` **plus the reference motion at that timestep**
(`joint_pos`, `joint_vel`, `body_pos_w[14,3]`, `body_quat_w[14,4]`, ...). The clip
is baked into the graph, so the robot does not need the npz.

Observation layout, in order, total 154, no history:

| Term | Dims | Source on hardware |
|---|---|---|
| `command` (ref `joint_pos` + `joint_vel`) | 58 | the ONNX's own reference outputs |
| `motion_anchor_ori_b` | 6 | torso orientation vs reference, see below |
| `base_ang_vel` | 3 | pelvis IMU gyro |
| `joint_pos` (relative to default pose) | 29 | encoders |
| `joint_vel` | 29 | encoders |
| `actions` | 29 | previous raw policy output |

`deploy.sh` runs the walking policy for teleop and switches to the dance on `x`.
The handoff is not instant: it settles, ramps the robot to the dance's first frame
over two seconds under position control, and only then hands over. Switching
directly would show the dance policy a large frame-0 tracking error and it would
lurch.

**Unverified before you run it:** whether the G1's `LowState_.imu_state.quaternion`
reports the *pelvis* or the *torso*. The tracking anchor is `torso_link`, so if it
is the pelvis, the three waist joint angles have to be composed onto it.
`./deploy.sh --probe-imu` decides this on hardware in 30 seconds by moving the
waist and watching whether the reported quaternion changes. Do that before the
first dance run.

## Open

- **Contact is still unvalidated.** Every identified parameter was measured with
  the base welded and limbs in the air. Tracking a dance stresses contact far
  harder than flat walking, and this is the largest untested part of the model.
- **The waist is nominal.** It is the only mechanical path between arms and legs
  and a dance moves it constantly. 15 minutes of hoisted robot time would fix it.
- `right_shoulder_pitch` had 3x the open-loop replay error of any other arm joint,
  unexplained.
- The 80 ms delay is set from the measured 60–80 ms hardware lag but has never
  been trained against on this task.
