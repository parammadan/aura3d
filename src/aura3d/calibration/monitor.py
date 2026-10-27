"""Calibration drift monitor (B3-cal): per-frame fixture-residual test against a commissioning baseline.

Commissioning: residuals on frames taken right after a verified calibration give mu0, sigma0.
Alarm rule (fixed before any test run): r > mu0 + k * sigma0, k = 6.
On alarm: fixture recalibration; accept only status "ok", else request operator (defer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aura3d.calibration import fixture as fx


@dataclass
class DriftMonitor:
    mu0: float
    sigma0: float
    k: float = 6.0
    events: list = field(default_factory=list)

    @classmethod
    def commission(cls, frames, layout, k: float = 6.0) -> "DriftMonitor":
        r = np.array([fx.residual(f, layout).median_abs_mm for f in frames])
        return cls(float(r.mean()), float(max(r.std(ddof=1), 1e-3)), k)

    @property
    def threshold(self) -> float:
        return self.mu0 + self.k * self.sigma0

    def check(self, frame, layout) -> tuple[bool, float]:
        r = fx.residual(frame, layout).median_abs_mm
        return r > self.threshold, r
