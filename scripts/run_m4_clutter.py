"""EXP-M4-002: does learned instance segmentation help the robot? B1 (classical) vs B2 (FT-A Mask R-CNN + same pose fit).

Scenes: UR10e, nominal noise, 6 parts (2 per class), min_sep 0.045 m so parts touch (the merged-blob case);
seeds 4000-4099 (test, never used in vision training: vision train seeds are 0-2999, val 100000+, test 200000+,
shift 300000+). Two measurements per arm:
  (1) perception: per true part, matched (<= 15 mm grasp-point error, right class) or not, on the first frame;
  (2) closed loop: sorting episode success (all 6 in their class bins), wrong picks.
Paired by seed. Run where both MuJoCo 3.17 and torch exist (cluster vision venv, GPU node).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aura3d.cameras.noise import PRESETS
from aura3d.controllers.episode import run_episode
from aura3d.evaluation.stats import mcnemar_exact, wilson
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.pose_estimation.learned import MaskRCNNLocalizer
from aura3d.simulation.env import WorkcellEnv, grasp_for
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seeds", default="4000:4100")
ap.add_argument("--out", default="results/m4/clutter.json")
ap.add_argument("--min-sep", type=float, default=0.045)
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = WorkcellEnv(WorkcellCfg(robot="ur10e", parts=["block", "pin", "bracket"] * 2), noise=PRESETS["nominal"])
mr = MaskRCNNLocalizer(a.ckpt)
from aura3d.controllers.episode import oracle_localizer  # noqa: E402
arms = {"B0": oracle_localizer(env), "B1": lambda f: detect_and_localize(f, env.meta), "B2": lambda f: mr(f, env.meta)}
top = env.meta["table_top"]
rows = []
for s in range(lo, hi):
    row = {"seed": s}
    for name, loc in arms.items():
        env.reset(s, min_sep=a.min_sep)
        f = env.render("fixed")
        ests = loc(f)
        matched = 0
        for i, c in enumerate(env.meta["part_classes"]):
            g = grasp_for(c, env.gt_part_pose(i), top)
            errs = [np.linalg.norm(grasp_for(e.features["cls"], e.T, top)["p"][:2] - g["p"][:2])
                    for e in ests if e.features["cls"] == c]
            matched += bool(errs) and min(errs) <= 0.015
        ep = run_episode(env, s, loc, max_attempts_per_part=2, reset=False, bin_slots=True, skip_failed=True)
        row[name] = {"n_dets": len(ests), "matched": matched, "correct": ep.correct, "success": ep.success,
                     "wrong_picks": sum(p["reason"] != "placed" for p in ep.picks)}
    rows.append(row)
    print(s, {k: (row[k]["matched"], row[k]["correct"]) for k in arms}, flush=True)
n = len(rows)
summ = {}
for k in arms:
    ks = sum(r[k]["success"] for r in rows)
    summ[k] = {"episode_success": ks / n, "wilson95": wilson(ks, n),
               "parts_matched_first_frame": sum(r[k]["matched"] for r in rows) / (6 * n),
               "parts_correct": sum(r[k]["correct"] for r in rows) / (6 * n),
               "wrong_picks": sum(r[k]["wrong_picks"] for r in rows)}
b = sum(r["B1"]["success"] and not r["B2"]["success"] for r in rows)   # B2 vs B1 (B0 = controller ceiling)
c = sum(r["B2"]["success"] and not r["B1"]["success"] for r in rows)
summ["mcnemar_p"] = mcnemar_exact(b, c)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))
