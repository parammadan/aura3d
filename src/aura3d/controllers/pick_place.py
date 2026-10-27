"""Deterministic geometric pick-and-place (the baseline every learned method is compared to).

Sequence: observe pose -> pre-grasp (+approach) -> descend -> close -> lift -> above bin
-> lower -> open -> retreat. The grasp target comes from the caller, so the same controller
serves the oracle baseline (B0, simulator pose) and perception baselines (B1+, estimated pose).
Grasp verification uses finger width only (proprioception).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PickPlaceCfg:
    approach: float = 0.15
    lift: float = 0.20
    place_height: float = 0.10     # TCP above table when releasing
    move_s: float = 1.2
    close_s: float = 0.6
    open_s: float = 0.4
    min_hold_width: float = 0.006
    width_tol: float = 0.015
    preshape_clearance: float = 0.025   # open to grasp width + this, not fully, so pads clear tray walls


@dataclass(frozen=True)
class PickPlaceResult:
    executed: bool
    grasp_verified: bool
    finger_width: float
    reason: str
    cycle_s: float


def pick_and_place(env, grasp: dict, bin_name: str, cfg: PickPlaceCfg = PickPlaceCfg(),  # noqa: B008
                   place_offset: tuple = (0.0, 0.0)) -> PickPlaceResult:
    robot, step = env.robot, env.step
    t0 = env.data.time
    top = env.meta["table_top"]
    T_g = robot.top_down(grasp["p"], grasp["yaw"])
    T_g2 = robot.top_down(grasp["p"], grasp["yaw"] + np.pi)   # parallel jaw: yaw and yaw+pi are equivalent
    pres = []
    for T in (T_g, T_g2):
        P = T.copy()
        P[2, 3] += cfg.approach
        pres.append(P)
    k, pre = robot.ik_best(pres)
    grasp_ik = robot.ik([T_g, T_g2][k], q0=pre.q)
    lift_T = pres[k].copy()
    lift_T[2, 3] = top + cfg.lift
    lift = robot.ik(lift_T, q0=grasp_ik.q)
    bc = env.meta["bins"][bin_name]["center"]
    place_T = robot.top_down(np.array([bc[0] + place_offset[0], bc[1] + place_offset[1], top + cfg.place_height]),
                             grasp["yaw"] + k * np.pi)
    above_T = place_T.copy()
    above_T[2, 3] = top + cfg.lift
    above = robot.ik(above_T, q0=lift.q)
    place = robot.ik(place_T, q0=above.q)
    if not all(r.success for r in (pre, grasp_ik, lift, above, place)):
        return PickPlaceResult(False, False, robot.gripper_width(), "ik_failed", env.data.time - t0)

    def wait(s):
        step(max(1, round(s / env.model.opt.timestep)))

    robot.command(width=grasp["width"] + cfg.preshape_clearance)
    if not (robot.move_to(pre.q, cfg.move_s, step) and robot.move_to(grasp_ik.q, cfg.move_s * 0.6, step)):
        return PickPlaceResult(True, False, robot.gripper_width(), "motion_blocked", env.data.time - t0)
    robot.command(width=0.0)
    wait(cfg.close_s)
    robot.move_to(lift.q, cfg.move_s * 0.6, step)
    w = robot.gripper_width()
    if w < cfg.min_hold_width:
        return PickPlaceResult(True, False, w, "closed_on_nothing", env.data.time - t0)
    if abs(w - grasp["width"]) > cfg.width_tol:
        return PickPlaceResult(True, False, w, "unexpected_width", env.data.time - t0)
    robot.move_to(above.q, cfg.move_s, step)
    robot.move_to(place.q, cfg.move_s * 0.6, step)
    robot.command(width=2 * 0.055)
    wait(cfg.open_s)
    robot.move_to(above.q, cfg.move_s * 0.6, step)
    wait(0.3)
    return PickPlaceResult(True, True, w, "placed", env.data.time - t0)
