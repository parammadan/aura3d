"""Build clips/<id>.mp4 for every narrated segment (1920x1080, 30 fps). Footage is real simulation video
time-fitted to its narration; cards carry numbers copied from results/ (see the comments next to each).
  ~/aura3d/.venv/bin/python make_clips.py [ids...]
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
HERE = Path(__file__).parent
DEMOS = HERE.parent / "demos"
BG, PANEL, TEXT, DIM = (11, 14, 20), (22, 27, 38), (232, 236, 242), (140, 150, 170)
CYAN, GREEN, RED, AMBER = (90, 200, 250), (110, 220, 140), (240, 110, 110), (250, 200, 90)


def F(sz, bold=True):
    for f in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else []) + [
            "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def dur(i):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "csv=p=0", str(HERE / "audio" / f"{i}.mp3")])) + 1.4


def base(kicker, title):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((120, 90), kicker.upper(), fill=CYAN, font=F(34))
    d.text((120, 140), title, fill=TEXT, font=F(64))
    return im, d


def bars(d, x, y, rows, vmax, w=860, h=70, gap=40, unit=""):
    for label, v, col in rows:
        d.text((x, y + 14), label, fill=TEXT, font=F(34, False))
        bx = x + 560
        d.rounded_rectangle([bx, y, bx + w, y + h], 10, fill=PANEL)
        d.rounded_rectangle([bx, y, bx + max(8, int(w * v / vmax)), y + h], 10, fill=col)
        d.text((bx + w + 30, y + 12), f"{v:g}{unit}", fill=TEXT, font=F(40))
        y += h + gap
    return y


def lines(d, x, y, items, sz=38, col=TEXT, gap=62):
    for t in items:
        c = AMBER if t.startswith("•") else col
        d.text((x, y), t, fill=c, font=F(sz, t.startswith("•")))
        y += gap
    return y


def foot(d, t):
    d.text((120, H - 90), t, fill=DIM, font=F(26, False))


def still(im, secs, out, zoom=1.04):
    n = int(secs * FPS)
    arr = np.array(im)
    wr = iio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    for k in range(n):                      # gentle Ken-Burns zoom so cards are not frozen (streamed: 8 GB laptop)
        z = 1 + (zoom - 1) * k / max(1, n - 1)
        cw, ch = int(W / z), int(H / z)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        wr.append_data(np.array(Image.fromarray(arr[y0:y0 + ch, x0:x0 + cw]).resize((W, H))))
    wr.close()


def footage(srcs, secs, out, labels=None, banner=None):
    readers = [iio.get_reader(s) for s in srcs]
    counts = [r.count_frames() for r in readers]
    n = int(secs * FPS)
    wr = iio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    for k in range(n):
        tiles = []
        for j, r in enumerate(readers):
            f = r.get_data(min(counts[j] - 1, int(k * counts[j] / n)))   # time-fit the whole clip, streamed
            tw = W // len(readers)
            im = Image.fromarray(f)
            s = min(tw / im.width, (H - 140) / im.height)
            im = im.resize((int(im.width * s), int(im.height * s)))
            tile = Image.new("RGB", (tw, H), BG)
            tile.paste(im, ((tw - im.width) // 2, 140 + (H - 140 - im.height) // 2))
            if labels:
                ImageDraw.Draw(tile).text((40, 150), labels[j], fill=TEXT, font=F(40))
            tiles.append(np.array(tile))
        fr_ = Image.fromarray(np.concatenate(tiles, 1))
        if banner:
            ImageDraw.Draw(fr_).text((120, 50), banner, fill=CYAN, font=F(46))
        wr.append_data(np.array(fr_))
    wr.close()


def make(i):
    out = str(HERE / "clips" / f"{i}.mp4")
    s = dur(i)
    if i == "s01":
        im, d = base("AURA-3D", "Half a degree is enough to break a pick cell")
        bars(d, 120, 380, [("No camera bump", 50, GREEN), ("Camera bumped 0.5°", 7, RED)], 50, unit=" / 50")
        foot(d, "Successful sorting episodes, perception only. Simulation (MuJoCo). results/m5/ANALYSIS.md")
        still(im, s, out)
    elif i == "s02":
        im, d = base("", "Param Madan")
        lines(d, 120, 300, ["Robot perception that knows when not to trust itself", "",
                            "github.com/parammadan/aura3d   ·   linkedin.com/in/parammadan"], sz=44, col=TEXT)
        still(im, s, out)
    elif i == "s03":
        footage([DEMOS / "m2_oracle_fanuc_m710ic_50.mp4", DEMOS / "m2_oracle_ur10e.mp4"], s, out,
                labels=["FANUC M-710iC/50 (official FANUC description)", "UR10e (MuJoCo Menagerie)"],
                banner="One workcell, two robots. Exact poses: 300/300 placed on each")
    elif i == "s04":
        im, d = base("Baseline", "Classical depth perception")
        lines(d, 120, 320, ["Depth image  →  3D points  →  segment each part  →  fit its CAD shape",
                            "", "• Nominal and degraded sensor noise: 100/100 episodes, both robots",
                            "• Noise alone cannot separate methods, so faults are injected:",
                            "    camera bumps, touching parts, distribution shift"])
        foot(d, "results/m3, docs/PERCEPTION.md")
        still(im, s, out)
    elif i == "s05":
        footage([DEMOS / "drift_b1_vs_aura.mp4"], s, out, banner="Same 0.5° camera bump. Left: perception only   Right: AURA-3D")
    elif i == "s06":
        im, d = base("EXP-M5-001 · 50 paired test episodes", "Camera bump 0.5°")
        y = bars(d, 120, 330, [("Perception only", 7, RED), ("+ monitor & checked recal", 42, GREEN)], 50, unit=" / 50")
        lines(d, 120, y + 20, ["• Wrong picks: 251 → 0      • False alarms with no bump: 0",
                               "• Unverifiable correction → stop and ask an operator (deferral)"])
        foot(d, "McNemar p = 6e-11. results/m5/ANALYSIS.md")
        still(im, s, out)
    elif i == "s07":
        im, d = base("Observability", "What a flat cell can't see")
        y = bars(d, 120, 330, [("10 mm bump, table + walls only", 7.5, RED), ("+ 3 reference spheres", 1.2, GREEN)], 8,
                 unit=" mm left")
        lines(d, 120, y + 20, ["• Condition number of JᵀJ did NOT flag bad fits",
                               "• Accept a fix only if it predicts a held-out sphere within 2 mm"])
        foot(d, "docs/CAMERA_CALIBRATION.md")
        still(im, s, out)
    elif i == "s08":
        im, d = base("EXP-M5-002 / 003", "Is this estimate good enough to grasp?")
        y = bars(d, 120, 330, [("Estimator's own confidence", 0.767, AMBER), ("Learned risk model", 0.966, GREEN)], 1,
                 unit=" AUROC")
        lines(d, 120, y + 20, ["• Measured tolerance: block 15, bracket 12, pin 10 mm",
                               "• Conformal 90 % bounds: 0.89 coverage in-distribution, 0.71 under shift"])
        foot(d, "AUROC on real-part detections. results/m5/RISK_ANALYSIS.md")
        still(im, s, out)
    elif i == "s09":
        im, d = base("EXP-M4-001", "Fine-tuning Mask R-CNN")
        y = bars(d, 120, 330, [("COCO weights, no fine-tune", 0.001, RED), ("Head-only fine-tune", 0.854, GREEN),
                               ("Full fine-tune", 0.861, CYAN)], 1, unit=" mAP")
        lines(d, 120, y + 20, ["• Head-only ≈ full within noise, at 28 % of the GPU memory (3.9 vs 13.9 GB)"])
        foot(d, "Segmentation mAP on 300 held-out synthetic images. docs/VISION_FINETUNING.md")
        still(im, s, out)
    elif i == "s10":
        footage([DEMOS / "clutter_ca.mp4"], s, out,
                banner="Example tray (seed 4204). Left: Mask R-CNN   Right: + least-crowded part first")
    elif i == "s11":
        im, d = base("EXP-M4-002/003 · M9-005", "Clutter, 100 new trays")
        y = bars(d, 120, 320, [("Classical depth", 30, RED), ("Fine-tuned Mask R-CNN", 53, AMBER),
                               ("+ least-crowded first", 88, GREEN), ("Exact poses + same rule", 88, CYAN)], 100, h=60,
                 gap=30, unit=" / 100")
        lines(d, 120, y + 10, ["• Risk gate + Mask R-CNN: wrong picks −49 %, success 21 → 31 / 50"])
        foot(d, "Fully sorted trays. results/m4/CLUTTER_NOTES.md, results/m9/NOTES.md")
        still(im, s, out)
    elif i == "s12":
        im, d = base("EXP-M9-002", "Same code on the FANUC M-710iC/50")
        y = bars(d, 120, 330, [("Perception only, 0.5° bump", 6, RED), ("+ monitor & checked recal", 42, GREEN)], 50,
                 unit=" / 50")
        lines(d, 120, y + 20, ["• Only change: --robot fanuc_m710ic_50 (config)",
                               "• Shared camera and cell: shows modular transfer, not a new cell"])
        foot(d, "McNemar p = 3e-11. results/m9_fanuc/NOTES.md")
        still(im, s, out)
    elif i == "s13":
        im, d = base("EXP-M6/M7/M8", "World models")
        y = bars(d, 120, 320, [("JEPA, zero-shot on our robot", 2.68, RED), ("JEPA, predictor fine-tuned", 0.67, AMBER),
                               ("Own latent model", 0.43, GREEN)], 3, h=60, gap=30, unit="×")
        lines(d, 120, y + 10, ["• Reproduced JEPA on its own task: 5/12 vs paper 41.6 %",
                               "• Grasp outcome from real frame: AUROC 0.99; from predicted frame: 0.72 → no control authority"])
        foot(d, "6-step prediction error vs 'nothing moves'. docs/WORLD_MODEL.md, results/m8/NOTES.md")
        still(im, s, out)
    elif i == "s14":
        im, d = base("Rigor and limits", "How the numbers were made")
        lines(d, 120, 320, [l for l in json.loads((HERE / "s14_lines.json").read_text())])
        foot(d, "EXPERIMENT_REGISTRY.md")
        still(im, s, out)
    elif i == "s15":
        im, d = base("The lesson", "Know when you're wrong. Have a good next move.")
        lines(d, 120, 340, ["github.com/parammadan/aura3d", "Code · results · pre-registered experiments · paper draft",
                            "", "linkedin.com/in/parammadan"], sz=46, gap=74)
        still(im, s, out)
    print(i, round(s, 1), "s")


if __name__ == "__main__":
    (HERE / "clips").mkdir(exist_ok=True)
    ids = sys.argv[1:] or [s["id"] for s in json.loads((HERE / "script.json").read_text()) if not s["text"].startswith("TBD")]
    for i in ids:
        make(i)
