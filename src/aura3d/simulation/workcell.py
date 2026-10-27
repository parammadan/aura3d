"""Build a MuJoCo industrial workcell for any robot in configs/robots.yaml.

World frame W: z up, origin at the robot base mount, table top at z = table_top.
Layout (all from config): table, pick tray, two place bins, a fixed camera, N parts.
The robot is loaded with MjSpec (MJCF or URDF), its own actuators replaced by
uniform position servos, gravity compensation enabled, and our parallel-jaw
gripper attached at the configured flange frame with prefix "robot/".

The gripper is ours (two slide-joint fingers, one servo + equality coupling). It is
deliberately simple so every robot carries the identical end effector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from aura3d.geometry import se3

ROOT = Path(__file__).resolve().parents[3]

# Part catalog: CAD-style primitives (the "CAD prior" perception may use).
# dims are half-sizes in m. "grasp_width" is the jaw width across the closing axis.
CATALOG = {
    "block": {"geoms": [("box", (0.030, 0.020, 0.015), (0, 0, 0))], "mass": 0.15, "rgba": (0.85, 0.35, 0.10, 1)},
    "pin": {"geoms": [("cylinder", (0.016, 0.025, 0), (0, 0, 0))], "mass": 0.10, "rgba": (0.55, 0.60, 0.65, 1)},
    "bracket": {
        "geoms": [("box", (0.035, 0.015, 0.006), (0, 0, -0.009)), ("box", (0.006, 0.015, 0.015), (-0.029, 0, 0))],
        "mass": 0.12, "rgba": (0.20, 0.45, 0.80, 1),
    },
}
GRIP_MAX_OPEN = 0.055     # m per finger -> 0.11 m max jaw width
TCP_OFFSET = 0.105        # m from flange to fingertip centre along tool z


def load_robot_cfg(name: str) -> dict:
    cfgs = yaml.safe_load((ROOT / "configs" / "robots.yaml").read_text())
    if name not in cfgs:
        raise KeyError(f"unknown robot {name!r}; have {sorted(cfgs)}")
    return cfgs[name]


@dataclass
class CameraCfg:
    name: str = "fixed"
    eye: tuple = (0.0, 0.0, 0.0)        # filled relative to table if left at zero
    target: tuple = (0.0, 0.0, 0.0)
    width: int = 640
    height: int = 400
    hfov_deg: float = 80.0              # OAK-D Pro FF stereo HFOV (datasheet), 1280x800 halved


@dataclass
class WorkcellCfg:
    robot: str = "ur10e"
    parts: list = field(default_factory=lambda: ["block", "pin", "bracket"])
    cameras: list = field(default_factory=lambda: [CameraCfg()])
    timestep: float = 0.002


def _rot_mj_from_cv(R_cv: np.ndarray) -> np.ndarray:
    """OpenCV camera (z fwd, y down) -> MuJoCo camera (looks along -z, y up)."""
    return R_cv @ np.diag([1.0, -1.0, -1.0])


def _add_position_servos(spec: mujoco.MjSpec, joints: list[str], kp, kv, prefix: str) -> None:
    for j, p, v in zip(joints, kp, kv):
        a = spec.add_actuator()
        a.name = f"{prefix}{j}"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = f"{prefix}{j}"
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = p
        a.biasprm[:3] = [0.0, -p, -v]
        a.ctrllimited = False


def _gripper_spec() -> mujoco.MjSpec:
    g = mujoco.MjSpec()
    base = g.worldbody.add_body(name="gripper")
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.07, 0.02], pos=[0, 0, 0.02],
                  rgba=[0.15, 0.15, 0.15, 1], mass=0.6)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        f = base.add_body(name=f"finger_{side}", pos=[0, 0, 0.04])
        f.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sgn, 0],
                    range=[0.0, GRIP_MAX_OPEN], damping=20.0, armature=0.01)
        f.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.006, 0.035],
                   pos=[0, sgn * 0.006, 0.035], rgba=[0.3, 0.3, 0.3, 1], mass=0.05,
                   friction=[1.5, 0.02, 0.002], condim=4)
    base.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004] * 3, rgba=[1, 0, 0, 0.5], group=4)
    eq = g.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_right", name2="finger_left")
    eq.data[:5] = [0, 1, 0, 0, 0]
    a = g.add_actuator(name="gripper", trntype=mujoco.mjtTrn.mjTRN_JOINT, target="finger_left")
    a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = 800.0
    a.biasprm[:3] = [0.0, -800.0, -40.0]
    a.forcerange = [-60.0, 60.0]       # N of squeeze per finger
    a.forcelimited = True
    a.ctrlrange = [0.0, GRIP_MAX_OPEN]
    a.ctrllimited = True
    return g


def _load_robot_spec(cfg: dict) -> mujoco.MjSpec:
    path = ROOT / cfg["path"]
    if cfg["source"] == "urdf":
        xml = path.read_text()
        hook = '<mujoco><compiler meshdir="{}" discardvisual="false" fusestatic="false"/></mujoco>'
        # inject compiler hints right after the opening <robot ...> tag
        i = xml.index(">", xml.index("<robot")) + 1
        xml = xml[:i] + hook.format(path.parent / "meshes") + xml[i:]
        spec = mujoco.MjSpec.from_string(xml)
    else:
        spec = mujoco.MjSpec.from_file(str(path))
    for k in list(spec.keys):
        spec.delete(k)
    if cfg.get("strip_actuators"):
        for a in list(spec.actuators):
            spec.delete(a)
    for j, arm in zip(cfg["joints"], cfg.get("armature", [None] * 6)):
        js = spec.joint(j)
        if arm is not None:
            js.armature = arm
        js.damping = np.maximum(np.asarray(js.damping), 1.0)
    return spec


def build_workcell(wc: WorkcellCfg) -> tuple[mujoco.MjModel, dict]:
    """Returns the compiled model and a dict of layout metadata (all world frame)."""
    rc = load_robot_cfg(wc.robot)
    spec = mujoco.MjSpec()
    spec.option.timestep = wc.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    wb = spec.worldbody
    top = float(rc["table_top"])
    cx, cy = rc["table_center"]

    wb.add_light(pos=[cx, cy, top + 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], pos=[0, 0, top - 0.75],
                rgba=[0.35, 0.35, 0.37, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.40, 0.55, 0.02], pos=[cx, cy, top - 0.02],
                rgba=[0.75, 0.75, 0.72, 1], friction=[1.0, 0.01, 0.001])
    # Tray: four low walls around the pick region.
    tray = {"center": np.array([cx, cy - 0.15, top]), "half": np.array([0.16, 0.16])}
    for k, (dx, dy, sx, sy) in enumerate([(0.16, 0, 0.005, 0.165), (-0.16, 0, 0.005, 0.165),
                                         (0, 0.16, 0.165, 0.005), (0, -0.16, 0.165, 0.005)]):
        wb.add_geom(name=f"tray_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.012],
                    pos=[tray["center"][0] + dx, tray["center"][1] + dy, top + 0.012], rgba=[0.3, 0.5, 0.3, 1])
    bins = {}
    for k, (name, dx) in enumerate([("bin_a", -0.12), ("bin_b", 0.12)]):
        c = np.array([cx + dx, cy + 0.30, top])
        bins[name] = {"center": c, "half": np.array([0.09, 0.09])}
        for j, (ox, oy, sx, sy) in enumerate([(0.09, 0, 0.005, 0.095), (-0.09, 0, 0.005, 0.095),
                                              (0, 0.09, 0.095, 0.005), (0, -0.09, 0.095, 0.005)]):
            wb.add_geom(name=f"{name}_wall{j}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.02],
                        pos=[c[0] + ox, c[1] + oy, top + 0.02], rgba=[0.6, 0.6, 0.2 + 0.4 * k, 1])

    cams = {}
    for c in wc.cameras:
        eye = np.array(c.eye) if np.any(c.eye) else np.array([cx + 0.55, cy - 0.15, top + 0.85])
        tgt = np.array(c.target) if np.any(c.target) else np.array([cx, cy, top])
        T_wc = se3.look_at(eye, tgt)
        q = se3.rot_to_quat(_rot_mj_from_cv(T_wc[:3, :3]))
        vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(c.hfov_deg) / 2) * c.height / c.width))
        wb.add_camera(name=f"cam/{c.name}", pos=eye, quat=q, fovy=vfov)
        fx = (c.width / 2) / np.tan(np.radians(c.hfov_deg) / 2)
        cams[c.name] = {"T_world_cam": T_wc, "width": c.width, "height": c.height,
                        "K": np.array([[fx, 0, (c.width - 1) / 2], [0, fx, (c.height - 1) / 2], [0, 0, 1]])}

    for i, cls in enumerate(wc.parts):
        cat = CATALOG[cls]
        b = wb.add_body(name=f"part/{i}_{cls}", pos=[cx, cy - 0.15, top + 0.2 + 0.05 * i])
        b.add_freejoint(name=f"part/{i}_{cls}")
        for k, (typ, size, pos) in enumerate(cat["geoms"]):
            gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[typ]
            b.add_geom(name=f"part/{i}_{cls}/g{k}", type=gtype, size=list(size), pos=list(pos), rgba=list(cat["rgba"]),
                       mass=cat["mass"] / len(cat["geoms"]), friction=[1.2, 0.02, 0.002], condim=4)

    # Robot + gripper
    robot = _load_robot_spec(rc)
    gripper = _gripper_spec()
    att = rc["attach"]
    if att["kind"] == "site":
        robot.site(att["name"]).attach_body(gripper.body("gripper"), "ee/", "")
    else:
        fr = robot.body(att["name"]).add_frame()
        fr.attach_body(gripper.body("gripper"), "ee/", "")
    mount = wb.add_frame(pos=rc["base_pos"])
    mount.attach_body(robot.worldbody.first_body(), "robot/", "")
    spec.compile()   # resolve names before adding servos that reference them
    # MuJoCo's parent-child filter is off for bodies welded to the world, so the base and
    # first link (overlapping convex hulls) would collide. Exclude every parent-child robot pair.
    for b in spec.bodies:
        par = b.parent
        if b.name.startswith("robot/") and par is not None and par.name.startswith("robot/"):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = par.name, b.name
    _add_position_servos(spec, rc["joints"], rc["kp"], rc["kv"], "robot/")
    for b in spec.bodies:
        if b.name.startswith("robot/"):
            b.gravcomp = 1.0
    model = spec.compile()
    meta = {"robot": wc.robot, "robot_cfg": rc, "table_top": top, "table_center": np.array([cx, cy]),
            "tray": tray, "bins": bins, "cameras": cams, "parts": [f"part/{i}_{c}" for i, c in enumerate(wc.parts)],
            "part_classes": list(wc.parts)}
    return model, meta
