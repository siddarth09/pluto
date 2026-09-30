#!/usr/bin/env bash
# Teleop the walking policy, press x to hand off to the Dance17 tracking policy.
#
#   ./deploy.sh --dry-run        # load both policies, publish nothing
#   ./deploy.sh --probe-imu      # decide --imu-frame, 30 s, waist only
#   ./deploy.sh                  # teleop + dance
#   ./deploy.sh --imu-frame torso
#
# deploy.py must run as a FILE, not -m: importing the package pulls in mjlab.
set -euo pipefail
exec env -u PYTHONPATH /home/sid/g1_real_env/bin/python \
  /home/sid/projects25/src/pluto/mjlab_g1/deploy.py --teleop "$@"
