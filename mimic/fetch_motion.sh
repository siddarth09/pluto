#!/usr/bin/env bash
# Fetch a clip from the exptech/g1-moves Space and resample it to 50 Hz.
#
#   ./fetch_motion.sh --list                  # every clip in the Space
#   ./fetch_motion.sh J_Dance17_Shuffle       # npz only, then prep
#   ./fetch_motion.sh B_SpinKarate --video    # also the capture/retarget mp4s
#
# The npz is already retargeted to the mjlab G1: 29 joints in SDK order, 30
# bodies pelvis..right_wrist_yaw_link, MuJoCo wxyz quaternions.
set -euo pipefail
SPACE=exptech/g1-moves
MIMIC=/home/sid/projects25/src/pluto/mimic
RAW=$MIMIC/../motions/g1-moves

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  echo "clips in $SPACE:"
  hf spaces list "$SPACE" --tree --recursive 2>/dev/null \
    | grep -oE '[A-Za-z0-9_]+\.npz' | sed 's/\.npz//' | sort -u | sed 's/^/  /'
  echo
  echo "usage: ./fetch_motion.sh <ClipName> [--video]"
  exit 0
fi

CLIP=$1; shift
PATTERN="media/**/$CLIP/training/$CLIP.npz"
[ "${1:-}" = "--video" ] && PATTERN="media/**/$CLIP/**"

echo "fetching $CLIP"
hf download "$SPACE" --type space --include "$PATTERN" --local-dir "$RAW" >/dev/null
SRC=$(find "$RAW" -name "$CLIP.npz" -print -quit)
[ -n "$SRC" ] || { echo "no $CLIP.npz in the Space -- try ./fetch_motion.sh --list"; exit 1; }

"$MIMIC/prep_motion.sh" "$SRC" "$MIMIC/motions/${CLIP}_50hz.npz"
echo
echo "train on it with:"
echo "  ./train.sh   # after pointing DEFAULT_MOTION in env_cfgs.py at ${CLIP}_50hz.npz"
