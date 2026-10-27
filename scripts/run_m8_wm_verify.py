"""EXP-M8-001: world-model grasp verification under camera drift.

Per seed (test 70000+, never used for trajectories): reset, bump the FIXED camera (rot deg, random axis; the wm
camera is untouched), localise with B1 on the fixed camera, take the largest detected blob. Arms, each from the
same reset state at the recorder's start pose:
  G     execute the scripted pick at the estimated grasp point (geometry only)
  WM    5x5 candidate grid (step 8 mm, +-16 mm) around the estimate, ranked by the fallback world model's
        P(success); DEFER if the best is < 0.5
  OR    oracle grasp point (reference)
Success = the targeted true part (nearest true part of the estimated class) is lifted >= 5 cm.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aura3d.cameras.noise import PRESETS
from aura3d.controllers.delta_policy import execute, go_to_start, make_wm_env, scripted_actions
from aura3d.evaluation.stats import mcnemar_exact, wilson
from aura3d.planning.wm_verifier import WMVerifier
from aura3d.pose_estimation.classical import detect_and_localize
from aura3d.simulation.env import grasp_for
from aura3d.simulation.faults import CameraDrift, apply_camera_drift

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seeds", default="70000:70100")
ap.add_argument("--rot", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
lo, hi = map(int, a.seeds.split(":"))
env = make_wm_env(noise=PRESETS["nominal"])
ver = WMVerifier(a.ckpt)
top = env.meta["table_top"]
grid = [np.array([dx, dy, 0.0]) for dx in (-0.016, -0.008, 0, 0.008, 0.016) for dy in (-0.016, -0.008, 0, 0.008, 0.016)]


def setup(seed):
    env.reset(seed)
    apply_camera_drift(env, CameraDrift(a.rot, 0.0, seed=seed))
    env.step()


rows = []
for s in range(lo, hi):
    setup(s)
    ests = detect_and_localize(env.render("fixed"), env.meta)
    if not ests:
        rows.append({"seed": s, "skipped": "no_detection"})
        continue
    est = max(ests, key=lambda e: e.features["pixels"])     # largest blob: phantoms are small (system-side rule)
    cls = est.features["cls"]
    g_est = grasp_for(cls, est.T, top)
    cand = [i for i, c in enumerate(env.meta["part_classes"]) if c == cls]
    tgt = min(cand, key=lambda i: np.linalg.norm(env.gt_part_pose(i)[:2, 3] - est.translation[:2]))
    g_true = grasp_for(cls, env.gt_part_pose(tgt), top)
    err_mm = float(np.linalg.norm(g_est["p"][:2] - g_true["p"][:2]) * 1000)
    row = {"seed": s, "cls": cls, "est_err_mm": err_mm}
    goals = {"G": g_est["p"], "OR": g_true["p"]}
    # WM: score candidates from the start pose
    setup(s)
    go_to_start(env, g_est["yaw"], g_est["width"])
    r = env.robot
    ch = ver.choose(env.render("wm", depth=False).rgb, r.tcp_pose()[:3, 3], r.gripper_width(),
                    [g_est["p"] + d for d in grid])
    row["wm_p_best"], row["wm_p_center"] = float(ch.p_success[ch.index]), float(ch.p_success[12])
    row["wm_choice_err_mm"] = float(np.linalg.norm((g_est["p"] + grid[ch.index])[:2] - g_true["p"][:2]) * 1000)
    row["wm_defer"] = ch.defer
    goals["WM"] = None if ch.defer else g_est["p"] + grid[ch.index]
    for arm, goal in goals.items():
        if goal is None:
            row[arm] = None
            continue
        setup(s)
        go_to_start(env, g_est["yaw"], g_est["width"])
        execute(env, scripted_actions(env.robot.tcp_pose()[:3, 3], goal), g_est["yaw"], g_est["width"])
        row[arm] = bool(env.gt_part_pose(tgt)[2, 3] > top + 0.05)
    rows.append(row)
    print(s, cls, round(err_mm, 1), {k: row[k] for k in ("G", "WM", "OR")}, "p", round(row["wm_p_best"], 2),
          round(row["wm_p_center"], 2), flush=True)

ok = [r for r in rows if "G" in r]
n = len(ok)
summ = {"rot_deg": a.rot, "n": n}
for arm in ("G", "WM", "OR"):
    k = sum(bool(r[arm]) for r in ok)
    summ[arm] = {"success": k / n, "wilson95": wilson(k, n), "deferred": sum(r[arm] is None for r in ok)}
summ["WM_wrong_actions"] = sum(r["WM"] is False for r in ok)
summ["G_wrong_actions"] = sum(r["G"] is False for r in ok)
b = sum(bool(r["G"]) and not bool(r["WM"]) for r in ok)
c = sum(bool(r["WM"]) and not bool(r["G"]) for r in ok)
summ["mcnemar_WM_vs_G"] = mcnemar_exact(b, c)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))
