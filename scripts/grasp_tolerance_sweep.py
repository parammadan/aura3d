"""EXP-M5-002: per-class grasp tolerance. Offset the ORACLE grasp point by delta (random xy direction) and/or
the jaw yaw, execute, and record success. tau_cls = largest offset whose success rate stays >= 95 %.

Dev seeds only (0..), so tau is fixed before the risk model is evaluated on test seeds.
  PYTHONPATH=src python scripts/grasp_tolerance_sweep.py --robot ur10e --seeds 0:40
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aura3d.controllers.pick_place import pick_and_place
from aura3d.evaluation.stats import wilson
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--seeds", default="0:40")
ap.add_argument("--offsets-mm", default="0,2,4,6,8,10,12,15")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot=a.robot))
res = {}
for d_mm in map(float, a.offsets_mm.split(",")):
    for i, cls in enumerate(env.meta["part_classes"]):
        ok = 0
        for s in range(lo, hi):
            env.reset(s)
            rng = np.random.default_rng([s, i, int(d_mm * 10)])
            g = env.grasp_world(i, env.gt_part_pose(i))
            th = rng.uniform(0, 2 * np.pi)
            g["p"] = g["p"] + d_mm / 1000 * np.array([np.cos(th), np.sin(th), 0])
            pick_and_place(env, g, "bin_a")
            ok += env.part_in_bin(i, "bin_a")
        n = hi - lo
        res.setdefault(cls, []).append({"offset_mm": d_mm, "success": ok / n, "wilson95": wilson(ok, n), "n": n})
        print(cls, d_mm, ok, "/", n, flush=True)
tau = {c: max([r["offset_mm"] for r in rs if r["success"] >= 0.95], default=0.0) for c, rs in res.items()}
Path("results/m5").mkdir(parents=True, exist_ok=True)
Path(f"results/m5/grasp_tolerance_{a.robot}.json").write_text(json.dumps({"sweep": res, "tau_mm": tau}, indent=1))
print("tau_mm", tau)
