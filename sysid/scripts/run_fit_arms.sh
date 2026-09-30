#!/usr/bin/env bash
# Arm fit: frictionloss pinned at the hardware-measured breakaway values,
# armature and damping fitted. Mirrors the shipping leg recipe. Friction is
# frozen because the fit provably cannot determine it -- on the legs it moved
# three orders of magnitude for under 3% change in predictive accuracy.
set -euo pipefail
SYSID_DIR=/home/sid/projects25/src/pluto/sysid
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=$SYSID_DIR/data
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=("$DATA/arm_chirp_0.pt" "$DATA/arm_chirp_1.pt")
fi
echo "=== datasets: ${DATASETS[*]}"

grep -q "^robot = 'g1_arms'" "$SYSID/config.py" || {
  echo "config.py is not set to g1_arms"; exit 1; }

echo "=== 1/2 install measured friction ==="
cd "$SYSID_DIR" && $PY scripts/set_measured_friction_arms.py

echo "=== 2/2 fit armature + damping, 28 free ==="
cd "$SYSID"
$PY -m sysid_mujoco.my_fit \
  --robot g1_arms \
  --dataset "${DATASETS[@]}" \
  --damping-bounds 0.001 10.0 \
  --armature-bounds 0.0001 0.5 \
  --freeze frictionloss --start-from-nominal \
  --optimizer scipy_parallel_fd \
  --max-iters 600

YAML=$(ls -t "$SYSID"/sysid_mujoco/results/g1_arms/*/opt_params.yaml | head -1)
echo "=== fitted params at $YAML"
cat "$YAML"
