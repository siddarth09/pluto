#!/usr/bin/env bash
# Play a trained policy in the viewer. Defaults to the newest checkpoint of the
# newest run, so no long paths to paste.
#
#   scripts: play.sh                       # newest run, newest checkpoint, flat-DR
#   play.sh 30000                          # a specific iteration
#   play.sh 32000 Pluto-Velocity-Rough-G1-Identified-DR
set -euo pipefail
SRC=/home/sid/projects25/src
PY="env -u PYTHONPATH PYTHONPATH=$SRC /home/sid/bheema_rl_env/bin/python"
ITER=${1:-}
TASK=${2:-Pluto-Velocity-Flat-G1-Identified-DR}

RUN=$(ls -td "$SRC"/logs/rsl_rl/pluto_g1_velocity_identified/*/ | head -1)
if [ -n "$ITER" ]; then
  CKPT="$RUN/model_${ITER}.pt"
  [ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; echo "available:"; ls "$RUN" | grep '^model_' | sort -t_ -k2 -n | tail -8; exit 1; }
else
  CKPT=$(ls "$RUN"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
fi

echo "run   : $RUN"
echo "ckpt  : $(basename "$CKPT")"
echo "task  : $TASK"
cd "$SRC"
$PY -m pluto.mjlab_g1.play "$TASK" --checkpoint-file "$CKPT"
