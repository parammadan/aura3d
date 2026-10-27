"""Record robot trajectories for world-model training (M7).

Action a_t = [dx, dy, dz, grip] in the world frame: TCP position delta in metres (|d| <= 0.02 per step) and
grip in {-1 (close), +1 (open)}. Control at 5 Hz (0.2 s per action; the servo tracks an IK target). Orientation
is held top-down with the yaw fixed at the episode's grasp yaw, so the action is 4-D like MetaWorld's Sawyer.
Policy: scripted pick-and-lift toward one part, with Gaussian action noise and, in a fraction of episodes, a
deliberately offset grasp, so the dataset holds successes and failures.

Saved per episode (npz, compressed): rgb (T+1, 224, 224, 3) uint8, tcp (T+1, 3), grip_w (T+1,), q (T+1, 6),
action (T, 4), success (bool: part lifted >= 5 cm at the end), part_cls, grasp_offset_mm, seed, cam (K, T_world_cam).
Splits by seed range: train 0.., val 50000.., test 60000.. (no shared layouts).
  PYTHONPATH=src python scripts/record_trajectories.py --split train --n 400 --out datasets/traj_v1
"""
import argparse
from pathlib import Path

import numpy as np

from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import CameraCfg, WorkcellCfg

SEED0 = {"train": 0, "val": 50_000, "test": 60_000}
STEP_S, DMAX, T = 0.2, 0.02, 24

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=list(SEED0), required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--out", default="datasets/traj_v1")
a = ap.parse_args()

wm_cam = CameraCfg(name="wm", eye=(0.0, 0.0, 0.0), width=224, height=224, hfov_deg=58.0)
env = WorkcellEnv(WorkcellCfg(robot=a.robot, cameras=[CameraCfg(), wm_cam]))
# wm camera: in front of the tray, looking at it from 0.9 m (overrides the default eye placement)
c = env.meta["table_center"]
top = env.meta["table_top"]
from aura3d.geometry import se3  # noqa: E402

T_wm = se3.look_at(np.array([c[0] + 0.75, c[1] - 0.15, top + 0.55]), np.array([c[0], c[1] - 0.15, top + 0.02]))
env.set_true_camera_pose(T_wm, "wm")
env.meta["cameras"]["wm"]["T_world_cam"] = T_wm
env.believed_T_world_cam["wm"] = T_wm
out = Path(a.out) / a.split
out.mkdir(parents=True, exist_ok=True)
r = env.robot

for k in range(a.n):
    seed = SEED0[a.split] + k
    rng = np.random.default_rng([seed, 31])
    env.reset(seed)
    i = int(rng.integers(len(env.meta["parts"])))
    g = env.grasp_world(i, env.gt_part_pose(i))       # scripted expert uses GT (data generation only)
    off = rng.uniform(8, 20) / 1000 if rng.random() < 0.35 else 0.0
    th = rng.uniform(0, 2 * np.pi)
    goal = g["p"] + off * np.array([np.cos(th), np.sin(th), 0])
    yaw = g["yaw"]
    # start pose: above the tray, top-down at this yaw
    start = r.top_down(env.meta["tray"]["center"] + np.array([0, 0, 0.22]), yaw)
    ik = r.ik_best([start])[1]
    r.set_joint_state(ik.q)
    r.command(width=g["width"] + 0.025)
    env.step(100)
    target = r.tcp_pose()[:3, 3].copy()
    waypoints = [goal + [0, 0, 0.12], goal, goal, goal + [0, 0, 0.15]]
    phase, grip = 0, 1.0
    rgb, tcp, gw, qs, acts = [], [], [], [], []

    def log():
        rgb.append(env.render("wm", depth=False).rgb)
        tcp.append(r.tcp_pose()[:3, 3].copy())
        gw.append(r.gripper_width())
        qs.append(r.q())

    log()
    for t in range(T):
        wp = waypoints[min(phase, 3)]
        d = np.clip(wp - target, -DMAX, DMAX) + rng.normal(0, 0.003, 3)
        d = np.clip(d, -DMAX, DMAX)
        if phase == 2:
            d, grip = np.zeros(3), -1.0
        if np.linalg.norm(wp - (target + d)) < 0.004 or phase == 2:
            phase += 1
        target = target + d
        ikr = r.ik(r.top_down(target, yaw), q0=r.q())
        r.command(q=ikr.q, width=(g["width"] + 0.025) if grip > 0 else 0.0)
        env.step(int(STEP_S / env.model.opt.timestep))
        acts.append([*d, grip])
        log()
    success = bool(env.gt_part_pose(i)[2, 3] > top + 0.05)
    np.savez_compressed(out / f"{seed:06d}.npz", rgb=np.array(rgb), tcp=np.array(tcp), grip_w=np.array(gw),
                        q=np.array(qs), action=np.array(acts, np.float32), success=success,
                        part_cls=env.meta["part_classes"][i], grasp_offset_mm=off * 1000, seed=seed,
                        K=env.meta["cameras"]["wm"]["K"], T_world_cam=T_wm)
    if k % 50 == 0:
        print(a.split, k, "success", success, "offset_mm", round(off * 1000, 1), flush=True)
