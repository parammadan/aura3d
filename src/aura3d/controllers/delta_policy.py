"""5 Hz Cartesian-delta controller shared by trajectory recording and world-model planning.

Action a = [dx, dy, dz, grip] (world frame, metres, |d| <= DMAX per 0.2 s step; grip +1 open, -1 close).
Orientation is held top-down at a fixed yaw. `scripted_actions` produces the noise-free pick-and-lift sequence
(approach above goal, descend, close, lift), the same phases as scripts/record_trajectories.py.
"""
from __future__ import annotations

import numpy as np

from aura3d.geometry import se3
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import CameraCfg, WorkcellCfg

STEP_S, DMAX, T = 0.2, 0.02, 24


def make_wm_env(robot: str = "ur10e", noise=None, **kw) -> WorkcellEnv:
    """Workcell with the fixed perception camera plus the 224x224 'wm' camera used to record traj_v*."""
    env = WorkcellEnv(WorkcellCfg(robot=robot, cameras=[CameraCfg(), CameraCfg(name="wm", width=224, height=224,
                                                                                    hfov_deg=58.0)], **kw), noise=noise)
    c, top = env.meta["table_center"], env.meta["table_top"]
    T_wm = se3.look_at(np.array([c[0] + 0.75, c[1] - 0.15, top + 0.55]), np.array([c[0], c[1] - 0.15, top + 0.02]))
    env.set_true_camera_pose(T_wm, "wm")
    env.meta["cameras"]["wm"]["T_world_cam"] = T_wm
    env.believed_T_world_cam["wm"] = T_wm
    return env


def go_to_start(env, yaw: float, width: float) -> None:
    r = env.robot
    start = r.top_down(env.meta["tray"]["center"] + np.array([0, 0, 0.22]), yaw)
    r.set_joint_state(r.ik_best([start])[1].q)
    r.command(width=width + 0.025)
    env.step(100)


def scripted_actions(start: np.ndarray, goal: np.ndarray, steps: int = T) -> np.ndarray:
    target, phase, out = np.asarray(start, float).copy(), 0, []
    wps = [goal + [0, 0, 0.12], goal, goal, goal + [0, 0, 0.15]]
    grip = 1.0
    for _ in range(steps):
        wp = wps[min(phase, 3)]
        d = np.clip(wp - target, -DMAX, DMAX)
        if phase == 2:
            d, grip = np.zeros(3), -1.0
        if np.linalg.norm(wp - (target + d)) < 0.004 or phase == 2:
            phase += 1
        target = target + d
        out.append([*d, grip])
    return np.array(out, np.float32)


def execute(env, actions: np.ndarray, yaw: float, width: float, on_step=None) -> None:
    r = env.robot
    target = r.tcp_pose()[:3, 3].copy()
    for a in actions:
        target = target + a[:3]
        ik = r.ik(r.top_down(target, yaw), q0=r.q())
        r.command(q=ik.q, width=(width + 0.025) if a[3] > 0 else 0.0)
        env.step(int(STEP_S / env.model.opt.timestep))
        if on_step is not None:
            on_step()
