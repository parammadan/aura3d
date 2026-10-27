"""Assemble the single AURA-3D demo video from real recorded clips plus text cards (numbers from results/).

  PYTHONPATH=src python scripts/make_full_demo.py --out ~/Desktop/videos/aura3d/AURA3D_demo.mp4
"""
import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 960, 368, 15
D = Path("demos")
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()


def font(sz):
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def card(title, lines=(), secs=3.0, sub=None):
    im = Image.new("RGB", (W, H), (14, 18, 26))
    d = ImageDraw.Draw(im)
    d.text((40, 36), title, fill=(255, 255, 255), font=font(34))
    y = 92
    if sub:
        d.text((40, 82), sub, fill=(150, 170, 200), font=font(17))
        y = 122
    for ln in lines:
        col = (255, 220, 120) if ln.startswith("•") else (220, 225, 235)
        d.text((40, y), ln, fill=col, font=font(20))
        y += 32
    return [np.array(im)] * int(secs * FPS)


def fit(frame, w, h):
    im = Image.fromarray(frame)
    s = min(w / im.width, h / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    bg = Image.new("RGB", (w, h), (0, 0, 0))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.array(bg)


def clip(path, speed=1.0, src_fps=15):
    r = iio.get_reader(path)
    step = max(1, round(speed * src_fps / FPS))
    return [fit(f, W, H) for i, f in enumerate(r) if i % step == 0]


def side_by_side(p1, p2, l1, l2, speed=3.0):
    a1, a2 = iio.get_reader(p1), iio.get_reader(p2)
    f1 = [f for i, f in enumerate(a1) if i % round(speed * 30 / FPS) == 0]
    f2 = [f for i, f in enumerate(a2) if i % round(speed * 30 / FPS) == 0]
    out = []
    for k in range(max(len(f1), len(f2))):
        L = Image.fromarray(fit(f1[min(k, len(f1) - 1)], W // 2, H))
        R = Image.fromarray(fit(f2[min(k, len(f2) - 1)], W // 2, H))
        for im, t in ((L, l1), (R, l2)):
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W // 2, 30], fill=(0, 0, 0))
            dr.text((8, 6), t, fill=(255, 255, 255), font=font(15))
        out.append(np.concatenate([np.array(L), np.array(R)], 1))
    return out


frames = []
frames += card("AURA-3D", ["Reliable industrial pick-and-place under changing conditions",
                           "Calibration-drift monitoring, pose-risk gating, learned vision, world models",
                           "", "Simulation (MuJoCo). Not hardware-validated. Not affiliated with any vendor."],
               secs=4, sub="Param Madan  |  github.com/parammadan/aura3d")
frames += card("1. Two robots, one workcell", ["FANUC M-710iC/50 from FANUC's official robot description",
                                                "UR10e from MuJoCo Menagerie", "Same gripper, camera, parts and code",
                                                "• Exact-pose control: 300/300 placed on each robot"], secs=3.5)
frames += side_by_side(D / "m2_oracle_fanuc_m710ic_50.mp4", D / "m2_oracle_ur10e.mp4",
                       "FANUC M-710iC/50", "UR10e", speed=3.0)
frames += card("2. The camera gets bumped 0.5°", ["Left: perception only. Pose estimates and phantom parts go wrong",
                                                   "Right: AURA-3D. Detects drift, recalibrates (checked on a held-out",
                                                   "reference sphere), gates risky picks, then sorts",
                                                   "• Over 50 test episodes: 7/50 → 42/50, wrong picks 251 → 0"], secs=4.5)
frames += clip(D / "drift_b1_vs_aura.mp4", speed=2.0)
frames += card("3. Touching parts (clutter)", ["Both use the fine-tuned Mask R-CNN (COCO-only: 0.001 mAP → 0.85)",
                                                "Right also picks the part with the most finger clearance first",
                                                "Example episode chosen where the rule matters. Aggregate below.",
                                                "• Over 100 test episodes: 50/100 → 88/100 (= exact poses + rule)"], secs=4.5)
frames += clip(D / "clutter_ca.mp4", speed=2.0)
frames += card("Results (held-out, paired, pre-registered)", [
    "• Camera bump 0.5°:  7/50 → 42/50 episodes;  FANUC, same code: 6/50 → 42/50",
    "• Clutter:  classical 30/100 → Mask R-CNN 53/100 → + clearance rule 88/100",
    "• Pose-risk model AUROC 0.97 vs 0.77;  risk gate + Mask R-CNN: 21 → 31/50",
    "• Meta JEPA world model: fine-tuning needed (2.7× → 0.67× copy error)",
    "• Neither world model foresees grasp outcome, so they get no control authority"], secs=8)
frames += card("Limits", ["Simulation only, assumed sensor noise, primitive parts, rigid camera bumps",
                          "JEPA-WMs is CC BY-NC (research only)",
                          "Next: real sensor data, gradual drift, informative recovery (re-view, singulation)"], secs=4,
               sub="github.com/parammadan/aura3d")
Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
iio.mimwrite(str(Path(a.out).expanduser()), frames, fps=FPS, quality=8, macro_block_size=8)
print("wrote", a.out, len(frames), "frames,", round(len(frames) / FPS), "s")
