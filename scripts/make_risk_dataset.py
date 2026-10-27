"""Build the pose-risk dataset: one row per detection = (features the system can see, true grasp-point error).

Per scene, conditions are drawn from the split's distribution:
  in-dist (train/cal/test): noise nominal|degraded, camera bump with p=0.5 (rot U[0,0.6] deg, trans U[0,5] mm),
                            clutter (touching parts) with p=0.3.
  shift:                    noise degraded only, bump always (rot U[0.3,1.5] deg, trans U[2,10] mm), clutter p=0.5.
Ground truth is used ONLY for the label: grasp-point xy error vs the nearest true part of the predicted class
(50 mm cap when there is none, e.g. a misclassified merged blob).
  PYTHONPATH=src python scripts/make_risk_dataset.py --split train --seeds 0:600
"""
import argparse
from pathlib import Path

import numpy as np

from aura3d.calibration import fixture as fx
from aura3d.cameras.noise import PRESETS
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.env import WorkcellEnv, grasp_for
from aura3d.simulation.faults import CameraDrift, apply_camera_drift
from aura3d.simulation.workcell import WorkcellCfg
from aura3d.uncertainty.risk import featurize

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "cal", "test", "shift"], required=True)
ap.add_argument("--seeds", required=True)
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--out", default="datasets/risk_v1")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
envs = {k: WorkcellEnv(WorkcellCfg(robot=a.robot), noise=PRESETS[k]) for k in ("nominal", "degraded")}
X, E, CLS, META = [], [], [], []
for s in range(lo, hi):
    rng = np.random.default_rng([s, 99])
    shift = a.split == "shift"
    nz = "degraded" if shift or rng.random() < 0.5 else "nominal"
    env = envs[nz]
    bump = shift or rng.random() < 0.5
    rot = rng.uniform(0.3, 1.5) if shift else (rng.uniform(0, 0.6) if bump else 0.0)
    tr = rng.uniform(2, 10) if shift else (rng.uniform(0, 5) if bump else 0.0)
    clutter = rng.random() < (0.5 if shift else 0.3)
    env.reset(s, min_sep=0.05 if clutter else 0.085)
    apply_camera_drift(env, CameraDrift(rot, tr, seed=s))
    env.step()
    f = env.render("fixed")
    r_cal = fx.residual(f, env.meta).median_abs_mm
    top = env.meta["table_top"]
    truth = [(c, grasp_for(c, env.gt_part_pose(i), top)) for i, c in enumerate(env.meta["part_classes"])]
    for est in detect_and_localize(f, env.meta):
        c = est.features["cls"]
        g = grasp_for(c, est.T, top)
        same = [gt for (ct, gt) in truth if ct == c]
        err = min((np.linalg.norm(g["p"][:2] - gt["p"][:2]) for gt in same), default=0.05)
        X.append(featurize(est, r_cal))
        E.append(min(err, 0.05) * 1000)          # mm
        CLS.append(c)
        META.append([s, rot, tr, clutter, nz == "degraded"])
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{a.split}_{lo}_{hi}.npz", X=np.array(X), e_mm=np.array(E), cls=np.array(CLS),
         meta=np.array(META, float), meta_cols=np.array(["seed", "rot_deg", "trans_mm", "clutter", "degraded"]))
print(a.split, len(E), "rows; error mm median", np.median(E).round(2), "p90", np.percentile(E, 90).round(2))
