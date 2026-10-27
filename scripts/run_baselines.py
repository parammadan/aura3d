"""Run closed-loop sorting episodes for a baseline on a seed range; write results/<exp>/<name>.json.

  PYTHONPATH=src python scripts/run_baselines.py --exp m3 --baseline B1 --robot ur10e --noise nominal
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from aura3d.calibration import fixture as fx
from aura3d.calibration.monitor import DriftMonitor
from aura3d.cameras.noise import PRESETS
from aura3d.simulation.faults import CameraDrift, apply_camera_drift
from aura3d.controllers.episode import oracle_localizer, run_episode
from aura3d.evaluation.stats import wilson
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--baseline", choices=["B0", "B1", "B3cal"], required=True)
ap.add_argument("--drift-rot", type=float, default=0.0, help="deg, sudden camera bump before the episode")
ap.add_argument("--drift-trans", type=float, default=0.0, help="mm")
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--noise", default="nominal", choices=list(PRESETS))
ap.add_argument("--seeds", default="1000:1100")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot=a.robot), noise=PRESETS[a.noise])
monitor = None
if a.baseline == "B3cal":   # commissioning on dev seeds 0..19, no drift
    frames = []
    for s in range(20):
        env.reset(s)
        frames.append(env.render("fixed"))
    monitor = DriftMonitor.commission(frames, env.meta)
cal_log = []


def b3_localize(frame):
    alarm, r = monitor.check(frame, env.meta)
    entry = {"t": frame.stamp, "residual_mm": r, "alarm": bool(alarm)}
    if alarm:
        rc = fx.recalibrate(frame, env.meta)
        entry.update({"recal": rc.status, "after_mm": rc.after_mm})
        cal_log.append(entry)
        if rc.status != "ok":
            return []                       # defer: no trustworthy geometry, operator needed
        env.believed_T_world_cam["fixed"] = rc.T_world_cam
        frame = env.render("fixed")
    else:
        cal_log.append(entry)
    return detect_and_localize(frame, env.meta)


loc = {"B0": oracle_localizer(env), "B1": lambda f: detect_and_localize(f, env.meta), "B3cal": b3_localize}[a.baseline]
t0, eps = time.time(), []
for s in range(lo, hi):
    drift = CameraDrift(a.drift_rot, a.drift_trans, seed=s)
    env.believed_T_world_cam["fixed"] = env.meta["cameras"]["fixed"]["T_world_cam"].copy()
    cal_log.clear()
    env.reset(s)
    apply_camera_drift(env, drift)
    r = run_episode(env, s, loc, reset=False)
    eps.append({"seed": s, "correct": r.correct, "n_parts": r.n_parts, "attempts": r.attempts, "success": r.success,
                "picks": r.picks, "end": r.reason, "fault": drift.to_dict(), "calib": list(cal_log)})
k_ep = sum(e["success"] for e in eps)
k_p, n_p = sum(e["correct"] for e in eps), sum(e["n_parts"] for e in eps)
summary = {"exp": a.exp, "baseline": a.baseline, "robot": a.robot, "noise": a.noise, "seeds": a.seeds,
           "drift_rot_deg": a.drift_rot, "drift_trans_mm": a.drift_trans,
           "monitor": None if monitor is None else {"mu0": monitor.mu0, "sigma0": monitor.sigma0, "thr": monitor.threshold},
           "alarm_episodes": sum(any(c["alarm"] for c in e["calib"]) for e in eps),
           "episodes": len(eps), "episode_success": k_ep / len(eps), "episode_wilson95": wilson(k_ep, len(eps)),
           "parts_correct": k_p, "parts": n_p, "part_rate": k_p / n_p, "part_wilson95": wilson(k_p, n_p),
           "mean_attempts": float(np.mean([e["attempts"] for e in eps])), "wall_s": time.time() - t0,
           "mujoco": mujoco.__version__,
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
out = Path("results") / a.exp
out.mkdir(parents=True, exist_ok=True)
name = f"{a.baseline}_{a.robot}_{a.noise}_r{a.drift_rot:g}_t{a.drift_trans:g}"
(out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": eps}, indent=1, default=float))
print(json.dumps(summary, default=float))
