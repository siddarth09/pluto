#!/usr/bin/env bash
# Play a trained mimic policy in the viewer. Defaults to the newest checkpoint
# of the newest run, so no long paths to paste.
#
#   ./play.sh            # newest run, newest checkpoint
#   ./play.sh 10000      # a specific iteration
set -euo pipefail
SRC=/home/sid/projects25/src
ITER=${1:-}
TASK=${2:-Pluto-Mimic-Dance17-G1-DR}

RUN=$(ls -td "$SRC"/logs/rsl_rl/pluto_g1_mimic_dance17/*/ | head -1)
if [ -n "$ITER" ]; then
  CKPT="$RUN/model_${ITER}.pt"
  [ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; echo "available:";
    ls "$RUN" | grep '^model_' | sort -t_ -k2 -n | tail -8; exit 1; }
else
  CKPT=$(ls "$RUN"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
fi
# mjlab's play.py re-resolves the motion file even though the env cfg sets it,
# and errors out with --checkpoint-file unless it is passed explicitly.
MOTION=${3:-$SRC/pluto/mimic/motions/J_Dance17_Shuffle_50hz.npz}
[ -f "$MOTION" ] || { echo "missing $MOTION -- run ./prep_motion.sh"; exit 1; }

echo "run   : $RUN"
echo "ckpt  : $(basename "$CKPT")"
echo "task  : $TASK"
echo "motion: $(basename "$MOTION")"
cd "$SRC"
exec env -u PYTHONPATH PYTHONPATH="$SRC" \
  /home/sid/bheema_rl_env/bin/python -m pluto.mimic.play "$TASK" \
  --checkpoint-file "$CKPT" --motion-file "$MOTION"
