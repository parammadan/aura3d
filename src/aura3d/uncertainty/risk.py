"""Pose-risk estimation: is this pose estimate good enough for THIS grasp? (master prompt §13)

Target (kept distinct from task failure): e = task-frame grasp-point error, the xy distance between the
grasp point computed from the estimate and from ground truth, plus the jaw-axis yaw error. A pick is
"out of tolerance" when e exceeds the per-class tolerance tau_cls measured in EXP-M5-002 (oracle grasp
offset sweep), not assumed.

Models:
  ConfidenceThreshold  baseline: abstain when the estimator's own quality score is low.
  GBRisk               gradient boosting on observation features -> predicted error; isotonic map of the
                       predicted error to P(out of tolerance) fitted on a calibration split.
  Conformal bound      split conformal (marginal) and Mondrian (per class) upper bound on e:
                       P(e <= e_hat + q) >= 1 - alpha holds *marginally over exchangeable data*, per class for
                       Mondrian. It is not a per-action guarantee and does not survive distribution shift; we
                       report empirical coverage under shift instead of assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from aura3d.contracts import PoseEstimate

FEATURES = ["rms_residual", "n_points", "pixels", "spread_xy", "valid_depth_frac", "quality", "calib_residual_mm",
            "cls_block", "cls_pin", "cls_bracket"]


def featurize(est: PoseEstimate, calib_residual_mm: float) -> np.ndarray:
    f = est.features
    c = f["cls"]
    return np.array([f["rms_residual"] * 1000, f["n_points"], f["pixels"], f["spread_xy"] * 1000,
                     f["valid_depth_frac"], est.quality, calib_residual_mm,
                     c == "block", c == "pin", c == "bracket"], float)


@dataclass
class GBRisk:
    alpha: float = 0.1
    model: HistGradientBoostingRegressor | None = None
    iso: IsotonicRegression | None = None
    q_marginal: float = np.nan
    q_class: dict | None = None

    def fit(self, X_tr, e_tr, X_cal, e_cal, cls_cal, tau_cal):
        # Fit on log error: errors span 0.1–20 mm and are right-skewed.
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   l2_regularization=1.0, random_state=0)
        self.model.fit(X_tr, np.log(e_tr + 1e-4))
        pred = self.predict_error(X_cal)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pred / tau_cal, e_cal > tau_cal)
        s = e_cal - pred                                        # nonconformity: under-prediction
        self.q_marginal = _conformal_q(s, self.alpha)
        self.q_class = {c: _conformal_q(s[cls_cal == c], self.alpha) for c in np.unique(cls_cal)}
        return self

    def predict_error(self, X) -> np.ndarray:
        return np.exp(self.model.predict(X)) - 1e-4

    def p_out(self, X, tau) -> np.ndarray:
        return self.iso.predict(self.predict_error(X) / tau)

    def upper(self, X, cls, mondrian: bool = True) -> np.ndarray:
        q = np.array([self.q_class[c] for c in cls]) if mondrian else self.q_marginal
        return self.predict_error(X) + q


def _conformal_q(s: np.ndarray, alpha: float) -> float:
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1]) if k <= n else np.inf


def risk_coverage(risk: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Accept lowest-risk first. Returns coverage, error rate among accepted, and the area under the curve."""
    o = np.argsort(risk)
    b = bad[o].astype(float)
    cov = np.arange(1, len(b) + 1) / len(b)
    err = np.cumsum(b) / np.arange(1, len(b) + 1)
    return cov, err, float(np.trapezoid(err, cov))
