"""EXP-M5-004: gradual drift detection delay + long-run false alarms for the DriftMonitor.

(a) False alarms: per seed, 20 observation frames with no drift (scene re-settled between frames by small robot
    motions) -> count alarms over seeds x frames.
(b) Gradual drift: per seed, the camera rotates linearly from 0 to `max_rot` degrees over `n_frames` observations
    (fixed random axis). Record the frame index and true magnitude at the first alarm, and the worst grasp-point
    error B1 would have made before that frame (perception run on each frame; GT used only for evaluation).
UR10e, nominal noise, monitor commissioned on dev seeds 0-19. Seeds 5000-5049 (fresh).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aura3d.calibration.monitor import DriftMonitor
from aura3d.cameras.noise import PRESETS
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.env import WorkcellEnv, grasp_for
from aura3d.simulation.faults import CameraDrift, apply_camera_drift
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="5000:5050")
ap.add_argument("--max-rot", type=float, default=1.0)
ap.add_argument("--n-frames", type=int, default=20)
ap.add_argument("--out", default="results/m5/gradual.json")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot="ur10e"), noise=PRESETS["nominal"])
fr = []
for s in range(20):
    env.reset(s)
    fr.append(env.render("fixed"))
mon = DriftMonitor.commission(fr, env.meta)
top = env.meta["table_top"]


def worst_err(frame):
    ests = detect_and_localize(frame, env.meta)
    errs = []
    for e in ests:
        c = e.features["cls"]
        g = grasp_for(c, e.T, top)["p"][:2]
        same = [grasp_for(c, env.gt_part_pose(i), top)["p"][:2] for i, k in enumerate(env.meta["part_classes"]) if k == c]
        errs.append(min(np.linalg.norm(g - t) for t in same) * 1000 if same else 50.0)
    return max(errs) if errs else 0.0


rows, fa_frames, fa_alarms = [], 0, 0
for s in range(lo, hi):
    env.reset(s)
    # reset() does not restore the camera; restore nominal explicitly (bug found in the first run, see registry)
    apply_camera_drift(env, CameraDrift(0.0, 0.0, seed=s))
    for k in range(a.n_frames):                       # (a) no drift
        env.step(25)
        fa_frames += 1
        fa_alarms += int(mon.check(env.render("fixed"), env.meta)[0])
    first = None
    errs = []
    for k in range(1, a.n_frames + 1):                # (b) ramp
        rot = a.max_rot * k / a.n_frames
        apply_camera_drift(env, CameraDrift(rot, 0.0, seed=s))
        env.step(25)
        f = env.render("fixed")
        alarm, r = mon.check(f, env.meta)
        if alarm and first is None:
            first = {"frame": k, "rot_deg": rot, "residual_mm": r}
            break
        errs.append(worst_err(f))
    rows.append({"seed": s, "first_alarm": first, "max_err_before_alarm_mm": max(errs) if errs else 0.0})
    print(s, first, round(rows[-1]["max_err_before_alarm_mm"], 1), flush=True)
det = [r["first_alarm"]["rot_deg"] for r in rows if r["first_alarm"]]
summ = {"false_alarm_frames": fa_frames, "false_alarms": fa_alarms, "detected": len(det), "n": len(rows),
        "rot_at_detection_deg": {"median": float(np.median(det)), "max": float(np.max(det))} if det else None,
        "max_err_before_alarm_mm": {"median": float(np.median([r["max_err_before_alarm_mm"] for r in rows])),
                                    "max": float(np.max([r["max_err_before_alarm_mm"] for r in rows]))},
        "ramp": {"max_rot_deg": a.max_rot, "frames": a.n_frames}, "monitor_thr_mm": mon.threshold}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))
