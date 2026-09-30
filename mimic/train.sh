#!/usr/bin/env bash
# Train the Dance17 Shuffle tracking policy.
#
#   ./train.sh                       # DR task, 4096 envs, video every 2k
#   ./train.sh Pluto-Mimic-Dance17-G1     # no-DR baseline, for debugging
#   ./train.sh Pluto-Mimic-Dance17-G1-DR 2048
set -euo pipefail
SRC=/home/sid/projects25/src
TASK=${1:-Pluto-Mimic-Dance17-G1-DR}
ENVS=${2:-4096}

MOTION=$SRC/pluto/mimic/motions/J_Dance17_Shuffle_50hz.npz
[ -f "$MOTION" ] || { echo "missing $MOTION -- run prep_motion.sh first"; exit 1; }

echo "task  : $TASK"
echo "envs  : $ENVS"
echo "motion: $(basename "$MOTION")"
cd "$SRC"
exec env -u PYTHONPATH PYTHONPATH="$SRC" MUJOCO_GL=egl \
  /home/sid/bheema_rl_env/bin/python -m pluto.mimic.train "$TASK" \
  --env.scene.num-envs "$ENVS" \
  --video True --video-interval 2000 --video-length 400
