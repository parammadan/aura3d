"""Map AURA-3D trajectories onto Meta JEPA-WM's MetaWorld interface (results/m6/JEPA_REPRO_REPORT.md).

JEPA-WM MetaWorld expects:
  image    uint8 (B, 1, 3, 224, 224), upright (MetaWorld's corner2 render is flipped by their env; ours is already
           upright, so no flip)
  proprio  [hand x, y, z (m, MetaWorld frame), gripper opening in [0, 1], 1 = open]
  action   per model step, 5 env sub-actions of [dx, dy, dz, grip] in [-1, 1]; xyz at 1 cm per unit;
           grip +1 = close (Sawyer: do_simulation([a, -a]); reset uses -1 = open). 20-D after concatenation.
AURA-3D (traj_v*): one 0.2 s step = Δxyz (m, |Δ| <= 0.02) and grip ±1 with +1 = OPEN.

Choices (documented; none is "the" right mapping):
  - one model step = one of our 0.2 s steps; our Δ is split evenly over the 5 sub-actions -> Δ/5 per sub-step in cm.
  - grip sign flipped.
  - proprio xyz re-centred: p_mw = proprio_mean_mw[:3] + (p_ours - c_ours), c_ours = our train-set mean TCP;
    no rotation between frames is attempted (embodiment and frames differ anyway).
  - jaw width / 0.11 m -> opening in [0, 1].
  - actions clipped to [-1, 1] BEFORE normalisation (the repro found unclipped ~6.6 sigma actions break prediction).
"""
from __future__ import annotations

import numpy as np

SUB = 5
MAX_JAW = 0.11


def adapt_actions(a: np.ndarray) -> np.ndarray:
    """(T, 4) ours -> (T, 5, 4) MetaWorld raw actions in [-1, 1]."""
    a = np.asarray(a, np.float32)
    xyz = np.clip(a[:, :3] * 100.0 / SUB, -1.0, 1.0)           # m -> cm, spread over 5 sub-steps
    grip = np.clip(-a[:, 3:4], -1.0, 1.0)                       # ours +1 open -> MW +1 close
    sub = np.concatenate([xyz, grip], 1)
    return np.repeat(sub[:, None, :], SUB, axis=1)


def adapt_proprio(tcp: np.ndarray, jaw: np.ndarray, c_ours: np.ndarray, mw_mean: np.ndarray) -> np.ndarray:
    """(T, 3) TCP + (T,) jaw width -> (T, 4) MetaWorld-style proprio."""
    xyz = np.asarray(mw_mean[:3]) + (np.asarray(tcp) - np.asarray(c_ours))
    return np.concatenate([xyz, np.clip(np.asarray(jaw) / MAX_JAW, 0, 1)[:, None]], 1).astype(np.float32)


def adapt_frames(rgb: np.ndarray) -> np.ndarray:
    """(T, 224, 224, 3) uint8 -> (T, 1, 3, 224, 224) uint8."""
    return np.ascontiguousarray(np.asarray(rgb, np.uint8).transpose(0, 3, 1, 2)[:, None])
