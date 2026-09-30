"""Prune g1_with_hands.xml to the arm-only sysid model.

Torso is harnessed and the waist is held at full gain, so the pelvis and legs are
rigid with the torso and share no dynamics with the arms. Re-roots at torso_link
and drops everything below the waist. Dex3 finger JOINTS go, the links stay: the
hand is mass at the wrist.

meshdir is inlined into every file=; _absolutize_file_attributes ignores
<compiler meshdir>.

    python scripts/make_arms_xml.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("/home/sid/projects25/src/bheema/unitree_g1")
DST = Path("/home/sid/projects25/src/sim2real-robot-identification/robot_model/g1_arms")

ARMS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw"]
JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ARMS]
# Motor spec reflected inertia. damping/frictionloss stay at the model default
# until the breakaway sweep measures them.
ARMATURE = {"shoulder_pitch": 0.00361, "shoulder_roll": 0.00361,
            "shoulder_yaw": 0.00361, "elbow": 0.00361, "wrist_roll": 0.00361,
            "wrist_pitch": 0.00425, "wrist_yaw": 0.00425}


def parent_map(root):
    return {c: p for p in root.iter() for c in p}


def main():
    tree = ET.parse(SRC / "g1_with_hands.xml")
    root = tree.getroot()
    root.set("model", "g1_arms")

    root.find("compiler").attrib.pop("meshdir", None)
    for mesh in root.find("asset").iter("mesh"):
        mesh.set("file", f"assets/{mesh.get('file')}")

    world = root.find("worldbody")
    pm = parent_map(root)
    torso = next(b for b in root.iter("body") if b.get("name") == "torso_link")

    # Re-root: torso is the only child of worldbody, on a freejoint for the
    # pipeline's fixed-base pass to strip.
    pm[torso].remove(torso)
    for b in list(world):
        world.remove(b)
    torso.set("pos", "0 0 1.0")
    torso.set("childclass", "g1")
    for j in torso.findall("joint"):
        torso.remove(j)
    torso.insert(0, ET.Element("freejoint", {"name": "floating_base_joint"}))
    world.append(torso)

    # Freeze the Dex3 fingers: mass and inertia stay, the DoF goes.
    for body in torso.iter("body"):
        for j in body.findall("joint"):
            if "_hand_" in j.get("name", ""):
                body.remove(j)

    # Per-joint armature: the fit starts from the motor spec, not the default.
    for body in torso.iter("body"):
        for j in body.findall("joint"):
            nm = j.get("name", "")
            for key, arm in ARMATURE.items():
                if nm.endswith(f"_{key}_joint"):
                    j.set("armature", f"{arm:.5f}")
                    j.set("damping", "0.5")
                    j.set("frictionloss", "0.3")

    # The pipeline rewrites existing actuators as `general`; it does not create
    # them. One position actuator per joint in config.Kp order, or
    # actuated_joint_names comes back empty.
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for c in list(act):
        act.remove(c)
    for nm in JOINTS:
        ET.SubElement(act, "position", {"class": "g1", "name": nm, "joint": nm})

    el = root.find("keyframe")
    if el is not None:
        root.remove(el)

    sensor = root.find("sensor")
    for c in list(sensor):
        sensor.remove(c)
    for nm in JOINTS:
        ET.SubElement(sensor, "jointpos", {"name": f"{nm}_pos", "joint": nm})

    # Drop meshes no surviving geom references.
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        nm = mesh.get("name") or Path(mesh.get("file")).stem
        if nm not in used:
            asset.remove(mesh)

    DST.mkdir(parents=True, exist_ok=True)
    if not (DST / "assets").exists():
        shutil.copytree(SRC / "assets", DST / "assets")
    ET.indent(tree, space="  ")
    tree.write(DST / "g1_arms.xml", encoding="utf-8", xml_declaration=True)
    # Frozen baseline, never fitted onto. set_measured_friction_arms.py and
    # apply_fit.py both build from it.
    tree.write(DST / "g1_arms_original.xml", encoding="utf-8", xml_declaration=True)
    print(f"wrote {DST / 'g1_arms.xml'} and g1_arms_original.xml")
    print(f"joints: {len(JOINTS)}  sensors: {len(sensor)}  actuators: {len(act)}")


if __name__ == "__main__":
    main()
