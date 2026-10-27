import numpy as np
import pytest

from aura3d.contracts import CameraFrame, PoseEstimate, RobotState


def test_pose_rejects_non_rotation():
    T = np.eye(4)
    T[0, 0] = 2.0
    with pytest.raises(ValueError):
        PoseEstimate(0.0, "p", "world", T, 1.0)


def test_pose_accessors():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    p = PoseEstimate(0.0, "p", "world", T, 1.0)
    assert np.allclose(p.translation, [0.1, 0.2, 0.3]) and np.allclose(p.rotation, np.eye(3))


def test_camera_frame_shape_check():
    with pytest.raises(ValueError):
        CameraFrame(0.0, "c", np.zeros((4, 5, 3), np.uint8), np.zeros((4, 4)), np.eye(3), np.eye(4))


def test_robot_state_shape_check():
    with pytest.raises(ValueError):
        RobotState(0.0, "r", np.zeros(6), np.zeros(5), np.eye(4), 0.0)
