#!/usr/bin/env bash
# Sweep one joint's actuator gain scale, fitting that joint's dynamics at each
# step with the residual masked to that joint. The minimum-cost scale is the
# gain the robot actually delivers -- independent of the tau_est regression,
# which the parallel ankle linkage makes unreliable.
#
#   scripts/scan_gain.sh left_knee            # default sweep
#   scripts/scan_gain.sh right_ankle_roll "0.6 0.8 1.0 1.2 1.4"
set -euo pipefail
SYSID_DIR=/home/sid/projects25/src/pluto/sysid
SYSID=/home/sid/projects25/src/sim2real-robot-identification
DATA=$SYSID_DIR/data
PY="env -u PYTHONPATH /home/sid/sysid_env/bin/python"
JOINT=${1:?usage: scan_gain.sh <joint-substring> [\"scales\"]}
SCALES=${2:-"0.70 0.80 0.85 0.90 0.95 1.00 1.05 1.15"}
BASE=${3:-chirp_1}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "scanning $JOINT on $BASE   scales: $SCALES"
printf "%8s  %s\n" "scale" "final cost"
for G in $SCALES; do
  $PY - "$DATA/$BASE.pt" "$TMP/s.pt" "$JOINT" "$G" <<'PYEOF'
import sys, torch, numpy as np
src, dst, joint, g = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
NAMES = ["left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee",
         "left_ankle_pitch","left_ankle_roll","right_hip_pitch","right_hip_roll",
         "right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll"]
idx = [i for i, n in enumerate(NAMES) if joint in n]
assert len(idx) == 1, f"{joint} matched {[NAMES[i] for i in idx]}"
d = torch.load(src, weights_only=False)
kp, kd = np.asarray(d["kp"]).copy(), np.asarray(d["kd"]).copy()
kp[idx[0]] *= g
kd[idx[0]] *= g
e = dict(d)
e["kp"] = torch.as_tensor(kp, dtype=torch.float32)
e["kd"] = torch.as_tensor(kd, dtype=torch.float32)
torch.save(e, dst)
PYEOF
  C=$(cd "$SYSID" && $PY -m sysid_mujoco.my_fit --dataset "$TMP/s.pt" \
        --fit-joints "$JOINT" --freeze frictionloss --start-from-nominal \
        --damping-bounds 0.001 10.0 --optimizer scipy_parallel_fd --max-iters 200 2>&1 \
      | grep -E "^Function evaluations" | sed -E 's/.*final cost ([0-9.e+-]+).*/\1/')
  printf "%8s  %s\n" "$G" "$C"
done
echo "lowest cost = the gain scale that joint actually delivers"
