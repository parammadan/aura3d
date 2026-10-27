"""EXP-M9-003: does the pose-risk gate add closed-loop value when geometry is fine but estimates are wrong?

Touching-part clutter (6 parts, min_sep 0.045), UR10e, nominal noise, no drift; seeds 4000-4049 (paired).
Both arms use AuraRunner with bin slots and max_fail_per_part=1; ONLY the risk gate differs (off vs on).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from aura3d.cameras.noise import PRESETS
from aura3d.controllers.episode import SORT_BINS
from aura3d.evaluation.stats import mcnemar_exact, wilson
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.recovery.state_machine import AuraRunner
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg
from aura3d.uncertainty.risk import GBRisk

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="4000:4050")
ap.add_argument("--out", default="results/m9/clutter_gate.json")
ap.add_argument("--cal", default="cal", help="calibration split glob prefix, e.g. cal_clutter6")
ap.add_argument("--localizer", choices=["b1", "b2"], default="b1")
ap.add_argument("--ckpt", default="checkpoints/maskrcnn_ftA/best.pt")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))


def load(split):
    ps = [np.load(f) for f in sorted(glob.glob(f"datasets/risk_v1/{split}_[0-9]*.npz"))]
    return {k: np.concatenate([p[k] for p in ps]) for k in ("X", "e_mm", "cls")}


tau = json.loads(Path("results/m5/grasp_tolerance_ur10e.json").read_text())["tau_mm"]
tr, cal = load("train"), load(a.cal)
risk = GBRisk().fit(tr["X"], tr["e_mm"], cal["X"], cal["e_mm"], cal["cls"], np.array([tau[c] for c in cal["cls"]]))
env = WorkcellEnv(WorkcellCfg(robot="ur10e", parts=["block", "pin", "bracket"] * 2), noise=PRESETS["nominal"])
if a.localizer == "b2":
    from aura3d.pose_estimation.learned import MaskRCNNLocalizer
    _mr = MaskRCNNLocalizer(a.ckpt)
    localize = lambda f: _mr(f, env.meta)  # noqa: E731
else:
    localize = lambda f: detect_and_localize(f, env.meta)  # noqa: E731
rows = []
for s in range(lo, hi):
    row = {"seed": s}
    for arm, rk in (("gate_off", None), ("gate_on", (risk, tau))):
        env.reset(s, min_sep=0.045)
        log = AuraRunner(env, localize, risk=rk, bin_slots=True).run()
        ok = sum(env.part_in_bin(i, SORT_BINS[c]) for i, c in enumerate(env.meta["part_classes"]))
        row[arm] = {"correct": int(ok), "success": ok == 6, "wrong": sum(p["reason"] != "placed" for p in log.picks),
                    "picks": len(log.picks), "outcome": log.outcome}
    rows.append(row)
    print(s, {k: (row[k]["correct"], row[k]["wrong"]) for k in ("gate_off", "gate_on")}, flush=True)
n = len(rows)
summ = {}
for arm in ("gate_off", "gate_on"):
    k = sum(r[arm]["success"] for r in rows)
    summ[arm] = {"episode_success": k / n, "wilson95": wilson(k, n), "parts_correct": sum(r[arm]["correct"] for r in rows),
                 "wrong_picks": sum(r[arm]["wrong"] for r in rows), "deferred": sum(r[arm]["outcome"] == "DEFER" for r in rows)}
b = sum(r["gate_off"]["success"] and not r["gate_on"]["success"] for r in rows)
c = sum(r["gate_on"]["success"] and not r["gate_off"]["success"] for r in rows)
summ["mcnemar_p"] = mcnemar_exact(b, c)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))
