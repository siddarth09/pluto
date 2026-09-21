"""Write the hardware-measured static friction into the model's frictionloss.

The values are the gravity torque each joint held without moving while hanging at
zero torque (README finding #6) -- a lower bound on breakaway friction, measured
with no estimator involved. Roll joints sat near gravity-neutral so their bound
was vacuous; they inherit the nearest informative figure.

Pair with `my_fit --freeze frictionloss --start-from-nominal` so the fit cannot
trade friction against damping.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SYSID = Path("/home/sid/projects25/src/sim2real-robot-identification")
MEASURED = {
    "left_hip_pitch_joint": 0.53, "left_hip_roll_joint": 0.59,
    "left_hip_yaw_joint": 0.16, "left_knee_joint": 0.67,
    "left_ankle_pitch_joint": 0.21, "left_ankle_roll_joint": 0.21,
    "right_hip_pitch_joint": 0.53, "right_hip_roll_joint": 0.59,
    "right_hip_yaw_joint": 0.16, "right_knee_joint": 0.85,
    "right_ankle_pitch_joint": 0.20, "right_ankle_roll_joint": 0.21,
}


def main() -> None:
    base = SYSID / "robot_model" / "g1_legs" / "g1_legs_original.xml"
    target = SYSID / "robot_model" / "g1_legs" / "g1_legs.xml"
    tree = ET.parse(base)
    joints = {j.get("name"): j for j in tree.getroot().iter("joint") if j.get("name")}
    for name, value in MEASURED.items():
        if name not in joints:
            raise SystemExit(f"{name} not in {base.name}")
        joints[name].set("frictionloss", f"{value:g}")
        print(f"  {name:<24} frictionloss {value:.2f}")
    shutil.copy(target, target.with_suffix(".xml.prev"))
    tree.write(target, encoding="utf-8", xml_declaration=True)
    print(f"\nwrote {target} (from the frozen baseline, armature/damping left nominal)")


if __name__ == "__main__":
    main()
