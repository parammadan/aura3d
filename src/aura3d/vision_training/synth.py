"""Synthetic instance-segmentation data from the workcell, with domain randomisation.

Labels come from MuJoCo's segmentation render (PRIVILEGED, labels only). Splits are separated by
seed range and by scene parameters, so no scene, layout or camera sample is shared between splits:
  train 0..N, val 100000.., test 200000.. (same distribution), shift 300000.. (held-out domain:
  unseen colour range, lighting and camera pose).
Clutter: a fraction of scenes use min_sep below the part size, so parts touch (the case where
depth connected components merge instances).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np

from aura3d.geometry import se3
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

CLASSES = ["block", "pin", "bracket"]          # category ids 1..3 (0 = background)
SPLIT_SEED0 = {"train": 0, "val": 100_000, "test": 200_000, "shift": 300_000}


@dataclass(frozen=True)
class DRCfg:
    hue_range: tuple = (0.0, 0.75)      # part colours drawn from this hue band (shift uses the rest)
    light_scale: tuple = (0.5, 1.1)
    cam_jitter_m: float = 0.03
    cam_jitter_deg: float = 3.0
    clutter_frac: float = 0.35
    eye_offset: tuple = (0.55, -0.15, 0.85)  # camera eye relative to table centre (shift: other side)


SHIFT = DRCfg(hue_range=(0.75, 1.0), light_scale=(1.1, 1.6), cam_jitter_m=0.03, cam_jitter_deg=3.0,
              clutter_frac=0.35, eye_offset=(-0.15, -0.6, 0.8))


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


class SceneGenerator:
    def __init__(self, robot: str = "ur10e"):
        self.env = WorkcellEnv(WorkcellCfg(robot=robot, parts=CLASSES * 2))
        m = self.env.model
        self.part_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b] for b in self.env.part_bodies]
        self.cam = self.env.cam_ids["fixed"]
        self.c0 = self.env.meta["table_center"]

    def scene(self, seed: int, dr: DRCfg):
        env, m = self.env, self.env.model
        rng = np.random.default_rng([seed, 5])
        clutter = rng.random() < dr.clutter_frac
        n_act = int(rng.integers(2, 7))
        while True:          # dense layouts can be infeasible at this spacing; drop a part rather than fail
            try:
                env.reset(seed, min_sep=0.045 if clutter else 0.085, n_active=n_act)
                break
            except RuntimeError:
                n_act -= 1
        for geoms in self.part_geoms:
            rgb = _hsv(rng.uniform(*dr.hue_range), rng.uniform(0.3, 0.9), rng.uniform(0.35, 0.95))
            for g in geoms:
                m.geom_rgba[g, :3] = rgb
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table"), :3] = rng.uniform(0.45, 0.85) * np.ones(3)
        m.light_diffuse[0] = rng.uniform(*dr.light_scale) * np.ones(3)
        m.light_pos[0] = [self.c0[0] + rng.uniform(-0.5, 0.5), self.c0[1] + rng.uniform(-0.5, 0.5), 2.5]
        top = env.meta["table_top"]
        eye = np.array([self.c0[0] + dr.eye_offset[0], self.c0[1] + dr.eye_offset[1], top + dr.eye_offset[2]])
        eye += rng.normal(0, dr.cam_jitter_m, 3)
        tgt = np.array([self.c0[0], self.c0[1] - 0.15, top]) + rng.normal(0, 0.02, 3)
        T = se3.look_at(eye, tgt)
        T[:3, :3] = T[:3, :3] @ se3.so3_exp(rng.normal(0, np.radians(dr.cam_jitter_deg) / np.sqrt(3), 3))
        env.set_true_camera_pose(T)
        f = env.render("fixed", depth=False)
        seg = env.render_segmentation("fixed")
        anns = []
        for i in env.active:
            mk = seg == i
            if mk.sum() < 30:
                continue                       # fully hidden / out of frame
            v, u = np.nonzero(mk)
            anns.append({"part": i, "category_id": CLASSES.index(env.meta["part_classes"][i]) + 1,
                         "bbox_xyxy": [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1],
                         "area": int(mk.sum())})
        inst = np.zeros(seg.shape, np.uint8)
        for k, a in enumerate(anns, start=1):
            inst[seg == a["part"]] = k
        return f.rgb, inst, anns, {"seed": seed, "clutter": bool(clutter), "T_world_cam": T.tolist()}


def write_split(out: Path, split: str, n: int, robot: str = "ur10e") -> None:
    dr = SHIFT if split == "shift" else DRCfg()
    gen = SceneGenerator(robot)
    d = out / split
    (d / "rgb").mkdir(parents=True, exist_ok=True)
    (d / "inst").mkdir(parents=True, exist_ok=True)
    index = []
    for k in range(n):
        seed = SPLIT_SEED0[split] + k
        rgb, inst, anns, meta = gen.scene(seed, dr)
        imageio.imwrite(d / "rgb" / f"{seed:06d}.png", rgb)
        imageio.imwrite(d / "inst" / f"{seed:06d}.png", inst)
        index.append({"file": f"{seed:06d}.png", "annotations": anns, **meta})
    (d / "index.json").write_text(json.dumps({"classes": CLASSES, "dr": asdict(dr), "robot": robot, "images": index}))
