#!/usr/bin/env bash
# Resample a g1-moves clip from 60 fps to the 50 Hz policy rate.
set -euo pipefail
MIMIC=/home/sid/projects25/src/pluto/mimic
SRC=${1:-$MIMIC/../motions/g1-moves/media/dance/J_Dance17_Shuffle/training/J_Dance17_Shuffle.npz}
DST=${2:-$MIMIC/motions/$(basename "${SRC%.npz}")_50hz.npz}
cd "$MIMIC"
exec env -u PYTHONPATH PYTHONPATH=/home/sid/projects25/src \
  /home/sid/bheema_rl_env/bin/python prep_motion.py "$SRC" "$DST" --output-fps 50
