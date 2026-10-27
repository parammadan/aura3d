"""Fault injection with explicit, seeded parameters (master prompt §20).

Every fault records type, parameters, seed, onset time and ground-truth severity so
experiment metadata can keep fault types separately identifiable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from aura3d.geometry import se3


@dataclass(frozen=True)
class CameraDrift:
    """True camera = nominal @ exp(drift); believed extrinsics unchanged.

    rot_deg / trans_mm are magnitudes; directions are drawn from `seed` (unit axis, unit vector).
    The perturbation is expressed in the camera frame (a bump to the mount)."""
    rot_deg: float = 0.0
    trans_mm: float = 0.0
    seed: int = 0
    kind: str = "sudden"        # "sudden" | "gradual"
    onset_s: float = 0.0

    def delta(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 77])
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        t = rng.normal(size=3)
        t /= np.linalg.norm(t)
        return se3.make_T(se3.so3_exp(a * np.radians(self.rot_deg)), t * self.trans_mm / 1000.0)

    def severity(self) -> dict:
        return {"rot_deg": self.rot_deg, "trans_mm": self.trans_mm}

    def to_dict(self) -> dict:
        return {"type": "camera_drift", **asdict(self)}


def apply_camera_drift(env, drift: CameraDrift, cam: str = "fixed") -> np.ndarray:
    """Move the true camera; returns the true T_world_cam."""
    T_true = env.meta["cameras"][cam]["T_world_cam"] @ drift.delta()
    env.set_true_camera_pose(T_true, cam)
    return T_true
