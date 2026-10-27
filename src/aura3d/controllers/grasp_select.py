"""Collision-aware grasp selection for clutter (perception-side: uses only estimated poses).

For a candidate grasp (point p, closing yaw, pre-shape width w), the two finger pads at the moment of descent
occupy boxes centred at p ± (w/2 + pad_half_thickness) along the closing axis, from the table up to the TCP.
Clearance = min signed distance from points sampled in those boxes to the catalog SDFs of all OTHER estimated parts.
  - pins are rotation-symmetric: search yaw over 0..pi for the largest clearance;
  - blocks/brackets have one closing axis (yaw and yaw + pi are equivalent);
  - pick order: largest clearance first (picking free parts first singulates the rest).
"""
from __future__ import annotations

import numpy as np

from aura3d.pose_estimation.classical import model_sdf
from aura3d.simulation.env import GRASPS, grasp_for

PAD_HALF = np.array([0.012, 0.006, 0.035])     # pad half-size (x along jaw, y across, z), from workcell._gripper_spec
PRESHAPE = 0.025


def _pad_points(p, yaw, width, n=5):
    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])      # closing axis
    t = np.array([-np.sin(yaw), np.cos(yaw), 0.0])     # along the jaw
    g = np.linspace(-1, 1, n)
    pts = []
    for side in (-1, 1):
        c = p + side * ((width + PRESHAPE) / 2 + PAD_HALF[1]) * a
        for u in g:
            for v in g:
                for w in np.linspace(0, 1, 3):
                    pts.append(c + u * PAD_HALF[0] * t + v * PAD_HALF[1] * a + np.array([0, 0, -0.005 + 0.06 * w]))
    return np.array(pts)


def clearance(grasp: dict, others: list) -> float:
    if not others:
        return 1.0
    P = _pad_points(grasp["p"], grasp["yaw"], grasp["width"])
    d = []
    for e in others:
        loc = (P - e.T[:3, 3]) @ e.T[:3, :3]
        d.append(model_sdf(e.features["cls"], loc).min())
    return float(min(d))


def best_grasp(est, others, table_top: float) -> tuple[dict, float]:
    cls = est.features["cls"]
    g = grasp_for(cls, est.T, table_top)
    if cls != "pin":
        return g, clearance(g, others)
    best = (g, clearance(g, others))
    for yaw in np.linspace(0, np.pi, 12, endpoint=False):
        h = dict(g, yaw=float(yaw))
        c = clearance(h, others)
        if c > best[1]:
            best = (h, c)
    return best


def choose(ests, table_top: float):
    """Return (estimate, grasp, clearance) with the largest clearance."""
    scored = []
    for k, e in enumerate(ests):
        g, c = best_grasp(e, ests[:k] + ests[k + 1:], table_top)
        scored.append((c, k, g))
    c, k, g = max(scored, key=lambda x: (x[0], -x[1]))
    return ests[k], g, c


__all__ = ["choose", "clearance", "best_grasp", "GRASPS"]
