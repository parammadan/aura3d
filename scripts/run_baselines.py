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

from aura3d.cameras.noise import PRESETS
from aura3d.controllers.episode import oracle_localizer, run_episode
from aura3d.evaluation.stats import wilson
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--baseline", choices=["B0", "B1"], required=True)
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--noise", default="nominal", choices=list(PRESETS))
ap.add_argument("--seeds", default="1000:1100")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot=a.robot), noise=PRESETS[a.noise])
loc = oracle_localizer(env) if a.baseline == "B0" else (lambda f: detect_and_localize(f, env.meta))
t0, eps = time.time(), []
for s in range(lo, hi):
    r = run_episode(env, s, loc)
    eps.append({"seed": s, "correct": r.correct, "n_parts": r.n_parts, "attempts": r.attempts, "success": r.success,
                "picks": r.picks, "end": r.reason})
k_ep = sum(e["success"] for e in eps)
k_p, n_p = sum(e["correct"] for e in eps), sum(e["n_parts"] for e in eps)
summary = {"exp": a.exp, "baseline": a.baseline, "robot": a.robot, "noise": a.noise, "seeds": a.seeds,
           "episodes": len(eps), "episode_success": k_ep / len(eps), "episode_wilson95": wilson(k_ep, len(eps)),
           "parts_correct": k_p, "parts": n_p, "part_rate": k_p / n_p, "part_wilson95": wilson(k_p, n_p),
           "mean_attempts": float(np.mean([e["attempts"] for e in eps])), "wall_s": time.time() - t0,
           "mujoco": mujoco.__version__,
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
out = Path("results") / a.exp
out.mkdir(parents=True, exist_ok=True)
name = f"{a.baseline}_{a.robot}_{a.noise}"
(out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": eps}, indent=1, default=float))
print(json.dumps(summary, default=float))
