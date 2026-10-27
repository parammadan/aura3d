"""Seeded RGB-D corruption for an active-stereo camera (applied to observations, never to ground truth).

Depth noise follows stereo triangulation: sigma_z = z^2 * sigma_d / (f_native * b), where b is the
baseline and f_native the focal length in pixels at the sensor's native resolution. Defaults use the
OAK-D Pro FF datasheet geometry (b = 75 mm, 1280 px wide, 80 deg HFOV -> f ~ 763 px).
sigma_d (sub-pixel disparity error) is an ASSUMED parameter, not fitted to any real unit: 0.1 px
"nominal", 0.3 px "degraded". Edge dropout mimics lost returns at depth discontinuities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoNoise:
    baseline_m: float = 0.075
    f_native_px: float = 640.0 / np.tan(np.radians(40.0))
    disparity_sigma_px: float = 0.1
    dropout: float = 0.005
    edge_jump_m: float = 0.03
    rgb_sigma: float = 2.0

    def depth_sigma(self, z: np.ndarray) -> np.ndarray:
        return z**2 * self.disparity_sigma_px / (self.f_native_px * self.baseline_m)


PRESETS = {
    "clean": None,
    "nominal": StereoNoise(),
    "degraded": StereoNoise(disparity_sigma_px=0.3, dropout=0.03, edge_jump_m=0.02, rgb_sigma=6.0),
}


def _edges(depth: np.ndarray, jump: float) -> np.ndarray:
    m = np.zeros(depth.shape, bool)
    dx = np.abs(np.diff(depth, axis=1)) > jump
    dy = np.abs(np.diff(depth, axis=0)) > jump
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    m[1:, :] |= dy
    m[:-1, :] |= dy
    return m


def corrupt(rgb: np.ndarray, depth: np.ndarray, cfg: StereoNoise, rng: np.random.Generator):
    if cfg.rgb_sigma > 0:
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, cfg.rgb_sigma, rgb.shape), 0, 255).astype(np.uint8)
    d = depth.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    d[valid] += (rng.standard_normal(valid.sum()) * cfg.depth_sigma(d[valid])).astype(np.float32)
    drop = _edges(depth, cfg.edge_jump_m) | (rng.random(d.shape) < cfg.dropout)
    d[drop | ~valid] = 0.0
    return rgb, d
