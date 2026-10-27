"""Typed data contracts shared by every module (master prompt §8).

Units: metres, radians, seconds, kilograms. Poses are 4x4 homogeneous T_AB mapping
frame-B coordinates into frame A (see aura3d.geometry.se3). Frame names are strings
such as "world", "robot_base", "camera/fixed", "object/<id>". Nothing here imports
the simulator, so perception and planning code can depend on it freely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _check_T(T: np.ndarray, name: str) -> None:
    T = np.asarray(T)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    R = T[:3, :3]
    if not (np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6):
        raise ValueError(f"{name} rotation block is not in SO(3)")


@dataclass(frozen=True)
class RobotState:
    stamp: float
    robot_id: str
    q: np.ndarray                 # rad, controller joint order
    qd: np.ndarray                # rad/s
    T_base_tcp: np.ndarray        # TCP in the robot base frame
    gripper_width: float          # m between fingers
    controller_status: str = "ok"

    def __post_init__(self):
        _check_T(self.T_base_tcp, "T_base_tcp")
        if self.q.shape != self.qd.shape:
            raise ValueError("q and qd must have the same shape")


@dataclass(frozen=True)
class CameraFrame:
    stamp: float
    frame_id: str
    rgb: np.ndarray               # (H, W, 3) uint8
    depth: np.ndarray | None      # (H, W) float32 z-depth in m; <= 0 or nan = no return
    K: np.ndarray                 # 3x3 intrinsics
    T_world_cam: np.ndarray       # extrinsics the system BELIEVES (may be wrong under drift)
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))   # OpenCV k1 k2 p1 p2 k3
    health: dict = field(default_factory=dict)

    def __post_init__(self):
        _check_T(self.T_world_cam, "T_world_cam")
        h, w = self.rgb.shape[:2]
        if self.depth is not None and self.depth.shape != (h, w):
            raise ValueError("depth and rgb must share resolution")


@dataclass(frozen=True)
class ObjectDetection:
    stamp: float
    object_id: str
    cls: str
    bbox_xyxy: np.ndarray         # pixels, float
    mask: np.ndarray | None       # (H, W) bool
    confidence: float


@dataclass(frozen=True)
class PoseEstimate:
    stamp: float
    object_id: str
    frame_id: str                 # frame the pose is expressed in
    T: np.ndarray                 # T_frame_object
    quality: float                # estimator's own score, higher is better
    cov6: np.ndarray | None = None   # 6x6 [rot(rad), trans(m)] tangent covariance, if known
    features: dict = field(default_factory=dict)   # inputs for the risk model

    def __post_init__(self):
        _check_T(self.T, "PoseEstimate.T")

    @property
    def translation(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.T[:3, :3]


class ActionType(str, Enum):
    MOVE_JOINT = "move_joint"
    MOVE_CARTESIAN = "move_cartesian"
    GRIPPER = "gripper"
    PICK = "pick"
    PLACE = "place"


@dataclass(frozen=True)
class Action:
    kind: ActionType
    q: np.ndarray | None = None           # joint target, rad
    T_target: np.ndarray | None = None    # Cartesian TCP target in world
    gripper_open: bool | None = None
    horizon_s: float = 1.0
    constraints: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelPrediction:
    latents: np.ndarray           # (H, ...) predicted future latents
    actions: np.ndarray           # (H, action_dim) candidate sequence, physical units documented by the adapter
    score: float                  # lower = better (planning cost)
    horizon: int
    uncertainty: float | None = None


class Decision(str, Enum):
    PROCEED = "proceed"
    REOBSERVE = "reobserve"
    RECALIBRATE = "recalibrate"
    DEFER = "defer"


@dataclass(frozen=True)
class ReliabilityDecision:
    decision: Decision
    risk: float                   # estimated P(out of tolerance) or task-frame error, see `risk_kind`
    risk_kind: str
    reasons: tuple[str, ...] = ()
