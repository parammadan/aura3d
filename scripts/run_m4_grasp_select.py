"""EXP-M4-003: collision-aware grasp selection in touching-part clutter.

Arms (paired by seed, 6 parts, min_sep 0.045, nominal noise, bin slots, skip_failed):
  B0 / B0+CA (oracle poses), B2 / B2+CA (FT-A Mask R-CNN). CA = controllers/grasp_select.choose over the current
  estimates (perception-side only). Seeds 4200-4299 (fresh).
"""
import argparse
import json
from pathlib import Path

from aura3d.cameras.noise import PRESETS
from aura3d.controllers.episode import oracle_localizer, run_episode
from aura3d.evaluation.stats import mcnemar_exact, wilson
from aura3d.pose_estimation.learned import MaskRCNNLocalizer
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seeds", default="4200:4300")
ap.add_argument("--out", default="results/m4/grasp_select.json")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot="ur10e", parts=["block", "pin", "bracket"] * 2), noise=PRESETS["nominal"])
mr = MaskRCNNLocalizer(a.ckpt)
locs = {"B0": oracle_localizer(env), "B2": lambda f: mr(f, env.meta)}
rows = []
for s in range(lo, hi):
    row = {"seed": s}
    for base, loc in locs.items():
        for ca in (False, True):
            env.reset(s, min_sep=0.045)
            r = run_episode(env, s, loc, reset=False, bin_slots=True, skip_failed=True, collision_aware=ca)
            row[base + ("+CA" if ca else "")] = {"correct": r.correct, "success": r.success,
                                                 "wrong": sum(p["reason"] != "placed" for p in r.picks)}
    rows.append(row)
    print(s, {k: v["correct"] for k, v in row.items() if k != "seed"}, flush=True)
n = len(rows)
summ = {}
for arm in ("B0", "B0+CA", "B2", "B2+CA"):
    k = sum(r[arm]["success"] for r in rows)
    summ[arm] = {"episode_success": k / n, "wilson95": wilson(k, n), "parts_correct": sum(r[arm]["correct"] for r in rows),
                 "wrong_picks": sum(r[arm]["wrong"] for r in rows)}
for base in ("B0", "B2"):
    b = sum(r[base]["success"] and not r[base + "+CA"]["success"] for r in rows)
    c = sum(r[base + "+CA"]["success"] and not r[base]["success"] for r in rows)
    summ[f"mcnemar_{base}"] = {"p": mcnemar_exact(b, c), "discordant": [b, c]}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))
