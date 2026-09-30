"""Write the measured static friction into the arm model's frictionloss.

Values from stiction_arms.py, gravity fed forward from live encoder positions.
Six of seven mirrored pairs agree; shoulder_roll does not and is installed at the
one direction that broke away, then widened in the randomisation.

Builds from g1_arms_original.xml. Run make_arms_xml.py first if the model changed.

Pair with `my_fit --freeze frictionloss --start-from-nominal`.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SYSID = Path("/home/sid/projects25/src/sim2real-robot-identification")
MEASURED = {
    "left_shoulder_pitch_joint": 0.65, "right_shoulder_pitch_joint": 0.60,
    "left_shoulder_roll_joint": 2.19, "right_shoulder_roll_joint": 1.81,
    "left_shoulder_yaw_joint": 0.50, "right_shoulder_yaw_joint": 0.52,
    "left_elbow_joint": 0.48, "right_elbow_joint": 0.61,
    "left_wrist_roll_joint": 0.46, "right_wrist_roll_joint": 0.46,
    "left_wrist_pitch_joint": 0.27, "right_wrist_pitch_joint": 0.31,
    "left_wrist_yaw_joint": 0.22, "right_wrist_yaw_joint": 0.20,
}
# Unresolved: breakaway exceeded the torque ceiling on one side.
UNRESOLVED = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")


def main() -> None:
    base = SYSID / "robot_model" / "g1_arms" / "g1_arms_original.xml"
    target = SYSID / "robot_model" / "g1_arms" / "g1_arms.xml"
    if not base.is_file():
        raise SystemExit(f"missing frozen baseline {base}; run make_arms_xml.py")
    tree = ET.parse(base)
    joints = {j.get("name"): j for j in tree.getroot().iter("joint") if j.get("name")}
    for name, value in MEASURED.items():
        if name not in joints:
            raise SystemExit(f"{name} not in {target.name}")
        joints[name].set("frictionloss", f"{value:g}")
        flag = "  UNRESOLVED, randomise wide" if name in UNRESOLVED else ""
        print(f"  {name:<28} frictionloss {value:.2f}{flag}")
    if target.is_file():
        shutil.copy(target, target.with_suffix(".xml.prev"))
    tree.write(target, encoding="utf-8", xml_declaration=True)
    print(f"\nwrote {target} (from the frozen baseline)")


if __name__ == "__main__":
    main()
