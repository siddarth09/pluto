#!/usr/bin/env bash
# Three arm models, same datasets, plots saved for each:
#   N  nominal everything (motor-spec armature/damping, frictionloss 0.3)
#   A  nominal armature/damping + hardware-measured friction
#   B  the fit's armature/damping + the same measured friction
# arm_chirp_* were in the fit; arm_shoulder_L / arm_elbow_L were not, so those
# are the holdout. A fit that only improves the training sets is rejected.
set -euo pipefail
SYSID_DIR=/home/sid/projects25/src/pluto/sysid
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=$SYSID_DIR/data
OUT=$SYSID_DIR/results/eval_arms
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"
SETS=(arm_chirp_0 arm_chirp_1 arm_shoulder_L arm_elbow_L)

YAML=${1:-$(ls -t "$SYSID"/sysid_mujoco/results/g1_arms/*/opt_params.yaml | head -1)}
echo "=== fit under test: $YAML"
rm -rf "$OUT"; mkdir -p "$OUT"

run_eval () {        # $1 = label, $2... = extra eval_fit flags
  local tag=$1; shift
  for D in "${SETS[@]}"; do
    [ -f "$DATA/$D.pt" ] || { printf "  %-16s MISSING\n" "$D"; continue; }
    printf "  %-16s " "$D"
    (cd "$SYSID" && $PY -m sysid_mujoco.eval_fit \
        --dataset "$DATA/$D.pt" --output-dir "$OUT/$tag/$D" --no-show "$@" 2>&1 \
      | grep -E "Mean RMSE" || echo "no RMSE line")
  done
}

echo "=== N: nominal everything ==="
run_eval N --original

echo "=== A: nominal armature/damping + measured friction ==="
cd "$SYSID_DIR" && $PY scripts/set_measured_friction_arms.py > /dev/null
run_eval A

echo "=== B: fitted armature/damping ==="
cd "$SYSID_DIR" && $PY scripts/apply_fit.py "$YAML" --robot g1_arms | tail -2
run_eval B

echo "=== restoring A ==="
cd "$SYSID_DIR" && $PY scripts/set_measured_friction_arms.py > /dev/null

$PY "$SYSID_DIR/scripts/eval_arms_index.py" "$OUT"
echo
echo "repeatability floor on this hardware: 0.0163 rad. Differences below that are noise."
