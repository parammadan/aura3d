"""Demo video: the same 0.5 deg camera bump, B1 (no monitoring) vs AURA-3D reliability loop, side by side.

Both panels are real simulation runs from the same seed. Captions show live state from each run.
  PYTHONPATH=src python scripts/record_drift_demo.py --seed 2003 --out demos/drift_b1_vs_aura.mp4
"""
import argparse
import glob
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from aura3d.calibration.monitor import DriftMonitor
from aura3d.cameras.noise import PRESETS
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.recovery.state_machine import AuraRunner, Budget
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.faults import CameraDrift, apply_camera_drift
from aura3d.simulation.workcell import WorkcellCfg
from aura3d.uncertainty.risk import GBRisk

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2003)
ap.add_argument("--rot", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
BINS = {"block": "bin_a", "bracket": "bin_a", "pin": "bin_b"}


def load(split):
    ps = [np.load(f) for f in sorted(glob.glob(f"datasets/risk_v1/{split}_[0-9]*.npz"))]
    return {k: np.concatenate([p[k] for p in ps]) for k in ("X", "e_mm", "cls")}


tau = json.loads(Path("results/m5/grasp_tolerance_ur10e.json").read_text())["tau_mm"]
tr, cal = load("train"), load("cal")
risk = GBRisk().fit(tr["X"], tr["e_mm"], cal["X"], cal["e_mm"], cal["cls"], np.array([tau[c] for c in cal["cls"]]))


def run(arm):
    env = WorkcellEnv(WorkcellCfg(robot="ur10e"), noise=PRESETS["nominal"])
    frames0 = []
    for s in range(20):
        env.reset(s)
        frames0.append(env.render("fixed"))
    mon = DriftMonitor.commission(frames0, env.meta)
    env.reset(a.seed)
    apply_camera_drift(env, CameraDrift(a.rot, 0.0, seed=a.seed))
    rend = mujoco.Renderer(env.model, 360, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [*env.meta["table_center"], 0.15]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 150, -30
    frames, status = [], {"text": "observing"}
    orig, cnt, every = env.step, [0], round((1 / 15) / env.model.opt.timestep)

    def step(n=1):
        for _ in range(n):
            orig()
            cnt[0] += 1
            if cnt[0] % every == 0:
                rend.update_scene(env.data, camera=cam)
                frames.append((rend.render().copy(), status["text"]))

    env.step = step
    mon_c = None if arm == "B1" else mon
    risk_c = None if arm == "B1" else (risk, tau)
    def caption(s0, s1, why):
        status["text"] = f"{s0} -> {s1}: {why}"[:52]

    # B1 panel: same runner with monitoring and risk off and retries allowed, i.e. the plain baseline loop
    runner = AuraRunner(env, lambda f: detect_and_localize(f, env.meta), monitor=mon_c, risk=risk_c,
                        budget=Budget(max_fail_per_part=2 if arm == "B1" else 1), on_transition=caption)
    log = runner.run()
    ok = sum(env.part_in_bin(i, BINS[c]) for i, c in enumerate(env.meta["part_classes"]))
    return frames, ok, log


panels = {}
for arm in ("B1", "AURA-3D"):
    panels[arm] = run("B1" if arm == "B1" else "AURA")
    print(arm, "sorted", panels[arm][1], "/ 3; picks", [p["reason"] for p in panels[arm][2].picks])
n = max(len(p[0]) for p in panels.values())
out = []
for k in range(n):
    row = []
    for arm, (frames, ok, _) in panels.items():
        img, txt = frames[min(k, len(frames) - 1)]
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 480, 44], fill=(0, 0, 0))
        d.text((8, 4), f"{arm}  |  camera bumped {a.rot} deg", fill=(255, 255, 255))
        d.text((8, 24), txt, fill=(255, 220, 120))
        if k >= len(frames) - 1:
            d.rectangle([0, 320, 480, 360], fill=(0, 0, 0))
            d.text((8, 330), f"result: {ok}/3 parts sorted correctly", fill=(120, 255, 120) if ok == 3 else (255, 120, 120))
        row.append(np.array(im))
    out.append(np.concatenate(row, 1))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimwrite(a.out, out, fps=15, quality=7)
print("wrote", a.out, len(out), "frames")
