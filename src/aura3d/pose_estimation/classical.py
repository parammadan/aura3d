"""Classical depth-only part localisation (baseline B1). No simulator state is read.

Inputs: a CameraFrame (with the extrinsics the system *believes*) and the workcell layout
prior (table height, tray region) plus the part catalog (CAD prior). Pipeline:
  depth -> world points -> crop to tray interior above the table -> connected components
  in the image -> class from height profile -> planar pose (x, y, yaw) by robust
  point-to-SDF least squares against the catalog model, multi-start over yaw.
Parts rest flat on the table, so z and roll/pitch are fixed by the class (a documented
assumption; tilted parts are a fault case, not handled here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from aura3d.contracts import CameraFrame, PoseEstimate
from aura3d.geometry import se3
from aura3d.geometry.camera import PinholeIntrinsics
from aura3d.simulation.workcell import CATALOG

HALF_HEIGHT = {"block": 0.015, "pin": 0.025, "bracket": 0.015}


@dataclass(frozen=True)
class ClassicalCfg:
    min_height: float = 0.004        # m above table to count as a part
    max_height: float = 0.09
    tray_margin: float = 0.008       # m inside tray walls
    min_pixels: int = 25
    huber: float = 0.002             # m, robust loss scale


def _sdf_box(p: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(p) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def _sdf_cyl(p: np.ndarray, r: float, h: float) -> np.ndarray:
    d = np.stack([np.linalg.norm(p[:, :2], axis=1) - r, np.abs(p[:, 2]) - h], axis=1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def model_sdf(cls: str, p_local: np.ndarray) -> np.ndarray:
    ds = []
    for typ, size, off in CATALOG[cls]["geoms"]:
        q = p_local - np.asarray(off)
        ds.append(_sdf_box(q, np.asarray(size)) if typ == "box" else _sdf_cyl(q, size[0], size[1]))
    return np.min(ds, axis=0)


def classify(heights: np.ndarray) -> str:
    """Height-profile rule (thresholds set from the catalog geometry, not tuned on test data)."""
    hmax = np.percentile(heights, 95)
    if hmax > 0.040:
        return "pin"
    low = np.mean(heights < 0.020)     # bracket: thin plate (12 mm) dominates the top view
    return "bracket" if low > 0.35 else "block"


def fit_planar_pose(cls: str, P: np.ndarray, z_table: float, huber: float = 0.002) -> tuple[np.ndarray, dict]:
    z = z_table + HALF_HEIGHT[cls]
    c0 = P[:, :2].mean(axis=0)
    xy = P[:, :2] - c0
    evals, evecs = np.linalg.eigh(xy.T @ xy)
    yaw0 = float(np.arctan2(evecs[1, 1], evecs[0, 1]))

    def res(x):
        R = se3.so3_exp([0, 0, x[2]])
        loc = (P - np.array([x[0], x[1], z])) @ R          # world -> part frame (R^T p)
        return model_sdf(cls, loc)

    best = None
    starts = [yaw0 + k * np.pi / 2 for k in range(4)] if cls != "pin" else [0.0]
    for y in starts:
        r = least_squares(res, [c0[0], c0[1], y], loss="huber", f_scale=huber, max_nfev=200)
        cost = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or cost < best[0]:
            best = (cost, r.x)
    cost, x = best
    T = se3.make_T(se3.so3_exp([0, 0, x[2]]), [x[0], x[1], z])
    return T, {"rms_residual": cost, "n_points": int(len(P)), "spread_xy": float(np.sqrt(evals.max() / len(P)))}


def detect_and_localize(frame: CameraFrame, layout: dict, cfg: ClassicalCfg = ClassicalCfg()) -> list[PoseEstimate]:  # noqa: B008
    if frame.depth is None:
        return []
    h, w = frame.depth.shape
    K = frame.K
    intr = PinholeIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h)
    v, u = np.mgrid[0:h, 0:w]
    d = frame.depth
    valid = np.isfinite(d) & (d > 0)
    Pc = intr.backproject(u[valid], v[valid], d[valid])
    Pw = se3.transform_points(frame.T_world_cam, Pc)
    top = layout["table_top"]
    tc, th = layout["tray"]["center"], layout["tray"]["half"] - cfg.tray_margin
    keep = ((Pw[:, 2] > top + cfg.min_height) & (Pw[:, 2] < top + cfg.max_height)
            & np.all(np.abs(Pw[:, :2] - tc[:2]) < th, axis=1))
    mask = np.zeros((h, w), bool)
    mask[v[valid][keep], u[valid][keep]] = True
    Pimg = np.full((h, w, 3), np.nan)
    Pimg[v[valid], u[valid]] = Pw
    labels, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        m = labels == k
        if m.sum() < cfg.min_pixels:
            continue
        P = Pimg[m]
        cls = classify(P[:, 2] - top)
        T, feats = fit_planar_pose(cls, P, top, cfg.huber)
        vv, uu = np.nonzero(m)
        feats.update({"pixels": int(m.sum()), "bbox": [int(uu.min()), int(vv.min()), int(uu.max()), int(vv.max())],
                      "valid_depth_frac": float(np.isfinite(d[m]).mean())})
        out.append(PoseEstimate(stamp=frame.stamp, object_id=f"det{k}:{cls}", frame_id="world", T=T,
                                quality=float(np.exp(-feats["rms_residual"] / 0.002)), features={"cls": cls, **feats}))
    return out
