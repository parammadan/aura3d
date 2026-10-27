"""EXP-M2-001: B0 oracle pick-and-place (simulator part poses) on held-out seeds, both robots.

Controller-only upper reference: perception is bypassed, so this measures kinematics,
servoing, gripper and task logic. Writes results/m2/b0_<robot>.json.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from aura3d.controllers.pick_place import pick_and_place
from aura3d.evaluation.stats import wilson
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--robot", required=True)
ap.add_argument("--seeds", default="1000:1100")
ap.add_argument("--out", default="results/m2")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot=a.robot))
rows, t0 = [], time.time()
for seed in range(lo, hi):
    env.reset(seed)
    for i, cls in enumerate(env.meta["part_classes"]):
        b = ("bin_a", "bin_b")[i % 2]
        r = pick_and_place(env, env.grasp_world(i, env.gt_part_pose(i)), b)   # ORACLE pose (privileged)
        rows.append({"seed": seed, "part": i, "cls": cls, "reason": r.reason, "placed": env.part_in_bin(i, b),
                     "cycle_s": r.cycle_s, "finger_width": r.finger_width})
k, n = sum(r["placed"] for r in rows), len(rows)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
summary = {"exp": "EXP-M2-001", "baseline": "B0_oracle", "robot": a.robot, "seeds": a.seeds, "n_picks": n,
           "placed": k, "rate": k / n, "wilson95": wilson(k, n),
           "cycle_s_mean": float(np.mean([r["cycle_s"] for r in rows])),
           "failures": [r for r in rows if not r["placed"]], "wall_s": time.time() - t0,
           "env": {"mujoco": mujoco.__version__, "python": platform.python_version(), "machine": platform.machine(),
                   "git": sha or "uncommitted"}}
Path(a.out).mkdir(parents=True, exist_ok=True)
Path(a.out, f"b0_{a.robot}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, default=float), "failures:", len(summary["failures"]))
