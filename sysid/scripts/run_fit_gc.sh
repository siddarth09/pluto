#!/usr/bin/env bash
# Refit with per-joint actuator gain correction applied, all 36 parameters free.
set -euo pipefail
SYSID_DIR=/home/sid/projects25/src/pluto/sysid
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=$SYSID_DIR/data
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"

echo "=== 1/4 reset model to pristine nominal ==="
cp "$SYSID/robot_model/g1_legs/g1_legs_original.xml" "$SYSID/robot_model/g1_legs/g1_legs.xml"

echo "=== 2/4 fit: all 36 free, gain-corrected data ==="
cd "$SYSID" && $PY -m sysid_mujoco.my_fit \
  --dataset "$DATA/chirp_1_gc.pt" "$DATA/chirp_1b_gc.pt" \
  --damping-bounds 0.001 10.0 --frictionloss-bounds 0.001 5.0 \
  --start-from-nominal --optimizer scipy_parallel_fd --max-iters 600

YAML=$(ls -t "$SYSID"/sysid_mujoco/results/g1_legs/*/opt_params.yaml | head -1)
echo "=== 3/4 install $YAML ==="
cd "$SYSID_DIR" && $PY scripts/apply_fit.py "$YAML"

echo "=== 4/4 evaluate (gain-corrected datasets) ==="
cd "$SYSID"
for D in chirp_1_gc chirp_0_gc; do
  printf "%-14s " "$D"
  $PY -m sysid_mujoco.eval_fit --dataset "$DATA/$D.pt" --no-show 2>&1 | grep -E "Mean RMSE"
done
