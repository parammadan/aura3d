"""EXP-M9-001: full reliability loop (AuraRunner) on EXP-M5-001 cells; paired with stored B1/B3cal results."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from aura3d.calibration.monitor import DriftMonitor
from aura3d.cameras.noise import PRESETS
from aura3d.evaluation.stats import mcnemar_exact, wilson
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.recovery.state_machine import AuraRunner
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.faults import CameraDrift, apply_camera_drift
from aura3d.simulation.workcell import WorkcellCfg
from aura3d.uncertainty.risk import GBRisk

ap = argparse.ArgumentParser()
ap.add_argument("--rot", type=float, required=True)
ap.add_argument("--trans", type=float, required=True)
ap.add_argument("--seeds", default="2000:2050")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))


def load(split):
    ps = [np.load(f) for f in sorted(glob.glob(f"datasets/risk_v1/{split}_*.npz"))]
    return {k: np.concatenate([p[k] for p in ps]) for k in ("X", "e_mm", "cls")}


tau = json.loads(Path("results/m5/grasp_tolerance_ur10e.json").read_text())["tau_mm"]
tr, cal = load("train"), load("cal")
risk = GBRisk().fit(tr["X"], tr["e_mm"], cal["X"], cal["e_mm"], cal["cls"], np.array([tau[c] for c in cal["cls"]]))
env = WorkcellEnv(WorkcellCfg(robot="ur10e"), noise=PRESETS["nominal"])
frames = []
for s in range(20):
    env.reset(s)
    frames.append(env.render("fixed"))
mon = DriftMonitor.commission(frames, env.meta)
eps = []
for s in range(lo, hi):
    env.believed_T_world_cam["fixed"] = env.meta["cameras"]["fixed"]["T_world_cam"].copy()
    env.reset(s)
    apply_camera_drift(env, CameraDrift(a.rot, a.trans, seed=s))
    log = AuraRunner(env, lambda f: detect_and_localize(f, env.meta), monitor=mon, risk=(risk, tau)).run()
    correct = sum(env.part_in_bin(i, {"block": "bin_a", "bracket": "bin_a", "pin": "bin_b"}[c])
                  for i, c in enumerate(env.meta["part_classes"]))
    eps.append({"seed": s, "correct": int(correct), "success": correct == 3, "outcome": log.outcome,
                "wrong_picks": sum(p["reason"] != "placed" for p in log.picks), "n_picks": len(log.picks),
                "transitions": log.transitions})
ref = {b: {e["seed"]: e for e in json.loads(Path(f"results/m5/{b}_ur10e_nominal_r{a.rot:g}_t{a.trans:g}.json")
                                            .read_text())["episodes"]} for b in ("B1", "B3cal")}
n = len(eps)
k6 = sum(e["success"] for e in eps)
summ = {"rot": a.rot, "trans": a.trans, "n": n, "B6": {"success": k6 / n, "wilson95": wilson(k6, n),
        "wrong_picks": sum(e["wrong_picks"] for e in eps), "deferred": sum(e["outcome"] == "DEFER" for e in eps)}}
for b, R in ref.items():
    kb = sum(R[e["seed"]]["success"] for e in eps)
    bb = sum(R[e["seed"]]["success"] and not e["success"] for e in eps)
    cc = sum(e["success"] and not R[e["seed"]]["success"] for e in eps)
    summ[f"vs_{b}"] = {b: kb / n, "mcnemar_p": mcnemar_exact(bb, cc)}
out = Path(f"results/m9/B6geo_r{a.rot:g}_t{a.trans:g}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"summary": summ, "episodes": eps}, indent=1, default=float))
print(json.dumps(summ, default=float))
