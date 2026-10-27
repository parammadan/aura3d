"""Demo clip: touching-part clutter, B2 (Mask R-CNN) vs B2 + collision-aware grasp selection, same seed, side by side.

Real runs. The live caption shows parts sorted so far and pick attempts. The seed is chosen from EXP-M4-003 as an
episode where the rule makes the difference; the end card states the aggregate result (50 vs 88 / 100).
  python scripts/record_clutter_demo.py --ckpt checkpoints/maskrcnn_ftA/best.pt --seed 4204 --out demos/clutter_ca.mp4
"""
import argparse

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from aura3d.cameras.noise import PRESETS
from aura3d.controllers import pick_place
from aura3d.controllers.episode import SORT_BINS, run_episode
from aura3d.pose_estimation.learned import MaskRCNNLocalizer
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seed", type=int, default=4204)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def run(ca: bool):
    env = WorkcellEnv(WorkcellCfg(robot="ur10e", parts=["block", "pin", "bracket"] * 2), noise=PRESETS["nominal"])
    mr = MaskRCNNLocalizer(a.ckpt)
    env.reset(a.seed, min_sep=0.045)
    rend = mujoco.Renderer(env.model, 360, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [env.meta["table_center"][0], env.meta["table_center"][1] + 0.05, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 150, -42
    frames, attempts = [], [0]
    orig, cnt, every = env.step, [0], round((1 / 15) / env.model.opt.timestep)

    def sorted_now():
        return sum(env.part_in_bin(i, SORT_BINS[c]) for i, c in enumerate(env.meta["part_classes"]))

    def step(n=1):
        for _ in range(n):
            orig()
            cnt[0] += 1
            if cnt[0] % every == 0:
                rend.update_scene(env.data, camera=cam)
                frames.append((rend.render().copy(), f"sorted {sorted_now()}/6   pick attempts {attempts[0]}"))

    env.step = step
    real_pp = pick_place.pick_and_place

    def counting_pp(*args, **kw):
        attempts[0] += 1
        return real_pp(*args, **kw)

    import aura3d.controllers.episode as ep
    ep.pick_and_place = counting_pp
    r = run_episode(env, a.seed, lambda f: mr(f, env.meta), reset=False, bin_slots=True, skip_failed=True,
                    collision_aware=ca)
    ep.pick_and_place = real_pp
    return frames, r.correct, attempts[0]


panels = {"Mask R-CNN only": run(False), "Mask R-CNN + least-crowded-first": run(True)}
for k, (_, ok, att) in panels.items():
    print(k, "sorted", ok, "/ 6 in", att, "attempts")
n = max(len(p[0]) for p in panels.values())
out = []
for k in range(n):
    row = []
    for name, (frames, ok, att) in panels.items():
        img, txt = frames[min(k, len(frames) - 1)]
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 480, 44], fill=(0, 0, 0))
        d.text((8, 4), f"{name}  |  6 touching parts", fill=(255, 255, 255))
        d.text((8, 24), txt, fill=(255, 220, 120))
        if k >= len(frames) - 1:
            d.rectangle([0, 316, 480, 360], fill=(0, 0, 0))
            d.text((8, 326), f"done: {ok}/6 sorted, {att} pick attempts", fill=(120, 255, 120) if ok == 6 else (255, 120, 120))
        row.append(np.array(im))
    out.append(np.concatenate(row, 1))
imageio.mimwrite(a.out, out, fps=15, quality=7, macro_block_size=8)
print("wrote", a.out, len(out))
