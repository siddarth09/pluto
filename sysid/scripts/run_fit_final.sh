#!/usr/bin/env bash
# Shipping model: per-joint actuator gain corrected, frictionloss pinned at the
# hardware-measured breakaway values, armature and damping fitted.
set -euo pipefail
SYSID_DIR=/home/sid/projects25/src/pluto/sysid
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=$SYSID_DIR/data
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"

echo "=== 1/4 install measured friction on a pristine model ==="
cd "$SYSID_DIR" && $PY scripts/set_measured_friction.py

echo "=== 2/4 fit armature + damping (24 free), gain-corrected data ==="
cd "$SYSID" && $PY -m sysid_mujoco.my_fit \
  --dataset "$DATA/chirp_1_gc.pt" "$DATA/chirp_1b_gc.pt" \
  --damping-bounds 0.001 10.0 \
  --freeze frictionloss --start-from-nominal \
  --optimizer scipy_parallel_fd --max-iters 600

YAML=$(ls -t "$SYSID"/sysid_mujoco/results/g1_legs/*/opt_params.yaml | head -1)
echo "=== 3/4 install $YAML ==="
cd "$SYSID_DIR" && $PY scripts/apply_fit.py "$YAML"

echo "=== 4/4 evaluate ==="
cd "$SYSID"
for D in chirp_1_gc chirp_0_gc; do
  printf "%-14s " "$D"
  $PY -m sysid_mujoco.eval_fit --dataset "$DATA/$D.pt" --no-show 2>&1 | grep -E "Mean RMSE"
done
