"""Calibration monitoring and correction from known static fixtures (table + tray/bin walls).

Residual r_fix: signed distance of observed static-scene points (mapped with the BELIEVED
extrinsics) to the workcell CAD boxes. Under correct calibration it is sensor noise only;
a camera bump produces a common-mode residual over the whole static scene.

Correction: robust least squares over a camera-frame SE(3) increment d (T_wc' = T_wc @ exp(d)),
with an observability report from the Gauss-Newton information matrix J^T J. Degenerate
directions (smallest eigenvalue below threshold) make the solver refuse rather than guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from aura3d.contracts import CameraFrame
from aura3d.geometry import se3
from aura3d.geometry.camera import PinholeIntrinsics


def _box_sdf(P, c, h):
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def _one_sdf(P, f):
    if f["kind"] == "box":
        return _box_sdf(P, f["center"], f["half"])
    q = P - f["center"]
    if f["kind"] == "sphere":
        return np.linalg.norm(q, axis=1) - f["radius"]
    d = np.stack([np.linalg.norm(q[:, :2], axis=1) - f["radius"], np.abs(q[:, 2]) - f["half_h"]], axis=1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def fixture_sdf(P: np.ndarray, fixtures: list[dict]) -> np.ndarray:
    return np.min([_one_sdf(P, f) for f in fixtures], axis=0)


def static_points(frame: CameraFrame, layout: dict, stride: int = 4) -> np.ndarray:
    """Camera-frame points from a subsampled depth image, excluding the tray interior where parts move."""
    d = frame.depth[::stride, ::stride]
    h, w = frame.depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    ok = np.isfinite(d) & (d > 0)
    K = frame.K
    intr = PinholeIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h)
    Pc = intr.backproject(u[ok], v[ok], d[ok])
    Pw = se3.transform_points(frame.T_world_cam, Pc)
    tc, th = layout["tray"]["center"], layout["tray"]["half"] - 0.012
    inside = np.all(np.abs(Pw[:, :2] - tc[:2]) < th, axis=1) & (Pw[:, 2] > layout["table_top"] + 0.003)
    tab = [f for f in layout["fixtures"] if f["name"] == "table"][0]
    near = np.all(np.abs(Pw[:, :2] - tab["center"][:2]) < tab["half"][:2] + 0.005, axis=1)   # table footprint only
    return Pc[~inside & near]


@dataclass(frozen=True)
class FixtureResidual:
    median_abs_mm: float
    mean_signed_mm: float
    n: int


def residual(frame: CameraFrame, layout: dict, T_world_cam: np.ndarray | None = None) -> FixtureResidual:
    Pc = static_points(frame, layout)
    T = frame.T_world_cam if T_world_cam is None else T_world_cam
    r = fixture_sdf(se3.transform_points(T, Pc), layout["fixtures"])
    r = np.clip(r, -0.05, 0.05)
    return FixtureResidual(float(np.median(np.abs(r)) * 1000), float(np.mean(r) * 1000), int(len(r)))


@dataclass(frozen=True)
class RecalResult:
    T_world_cam: np.ndarray
    converged: bool
    observable: bool
    min_eig: float
    eigs: np.ndarray
    before_mm: float
    after_mm: float
    status: str
    holdout_mm: tuple = ()


def _fit(Pc, T0, fixtures, huber_m):
    def res(x):
        T = T0 @ se3.make_T(se3.so3_exp(x[:3]), x[3:])
        return fixture_sdf(se3.transform_points(T, Pc), fixtures)
    sol = least_squares(res, np.zeros(6), loss="huber", f_scale=huber_m, x_scale=[0.01] * 6, max_nfev=200)
    return sol, res


def _sphere_owner(P_w: np.ndarray, fixtures: list[dict], tol: float = 0.02) -> np.ndarray:
    """Index of the reference sphere each world point lies near (-1 if none)."""
    own = np.full(len(P_w), -1)
    for k, f in enumerate(fx for fx in fixtures if fx["kind"] == "sphere"):
        d = np.abs(np.linalg.norm(P_w - f["center"], axis=1) - f["radius"])
        own[(d < tol) & (own < 0)] = k
    return own


def recalibrate(frame: CameraFrame, layout: dict, huber_m: float = 0.003, min_eig_ratio: float = 1e-4,
                max_correction_deg: float = 10.0, max_correction_mm: float = 100.0, val_tol_mm: float = 2.0,
                min_val_points: int = 20) -> RecalResult:
    """Fit on all static points; validate by leave-one-sphere-out refits (each held-out sphere must be
    predicted within val_tol_mm). Without visible spheres the result is 'unvalidated' and not applied."""
    Pc = static_points(frame, layout, stride=3)
    T0 = frame.T_world_cam
    fixtures = layout["fixtures"]
    sol, res = _fit(Pc, T0, fixtures, huber_m)
    before = float(np.median(np.abs(res(np.zeros(6))))) * 1000
    J = sol.jac
    # Information per unit of (rad, m) scaled to (0.01 rad, 0.01 m) so eigenvalues are comparable.
    Js = J * 0.01
    eigs = np.linalg.eigvalsh(Js.T @ Js / len(Pc))
    # Numerical sanity only (condition < 1e4). Measured: NOT discriminative between good and bad fits
    # (docs/CAMERA_CALIBRATION.md), so acceptance rests on the held-out sphere validation below.
    observable = bool(eigs.min() / eigs.max() > min_eig_ratio)
    T1 = T0 @ se3.make_T(se3.so3_exp(sol.x[:3]), sol.x[3:])
    after = float(np.median(np.abs(sol.fun))) * 1000
    too_big = np.degrees(np.linalg.norm(sol.x[:3])) > max_correction_deg or np.linalg.norm(sol.x[3:]) * 1000 > max_correction_mm
    # Leave-one-sphere-out validation.
    owner = _sphere_owner(se3.transform_points(T1, Pc), fixtures)
    spheres = [f for f in fixtures if f["kind"] == "sphere"]
    val = []
    for k, sph in enumerate(spheres):
        held = owner == k
        if held.sum() < min_val_points:
            continue
        s_k, _ = _fit(Pc[~held], T0, fixtures, huber_m)
        Tk = T0 @ se3.make_T(se3.so3_exp(s_k.x[:3]), s_k.x[3:])
        r = np.linalg.norm(se3.transform_points(Tk, Pc[held]) - sph["center"], axis=1) - sph["radius"]
        val.append(float(np.median(np.abs(r))) * 1000)
    if not observable:
        status = "degenerate"
    elif too_big:
        status = "implausible"
    elif not (sol.success and after < before):
        status = "no_improvement"
    elif len(val) < 2:
        status = "unvalidated"
    elif max(val) > val_tol_mm:
        status = "validation_failed"
    else:
        status = "ok"
    return RecalResult(T1 if status == "ok" else T0, bool(sol.success), observable, float(eigs.min()), eigs,
                       before, after, status, tuple(val))
