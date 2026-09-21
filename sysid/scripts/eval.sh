#!/usr/bin/env bash
# Open-loop replay of a dataset against the CURRENT model in robot_model/g1_legs/.
#   scripts/eval.sh chirp_0_gc            # current model
#   scripts/eval.sh chirp_0_gc --nominal  # reset model to pristine first
set -euo pipefail
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=/home/sid/projects25/src/pluto/sysid/data
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"
STEM=${1:?usage: eval.sh <dataset-stem> [--nominal] [--show]}
SHOW="--no-show"
for a in "$@"; do [ "$a" = "--show" ] && SHOW="--show"; done

case " $* " in *" --nominal "*)
  cp "$SYSID/robot_model/g1_legs/g1_legs_original.xml" "$SYSID/robot_model/g1_legs/g1_legs.xml"
  echo "model reset to pristine nominal (armature 0.01, frictionloss 0.3, damping 0)";;
esac

cd "$SYSID"
$PY -m sysid_mujoco.eval_fit --dataset "$DATA/$STEM.pt" $SHOW 2>&1 \
  | grep -E "Mean RMSE|Mean MAE|Saved trajectory plot"
