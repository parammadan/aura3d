"""M2 gates: both embodiments compile, track, solve IK, reset reproducibly and pick-and-place (oracle)."""
import numpy as np
import pytest

from aura3d.controllers.pick_place import pick_and_place
from aura3d.geometry import se3
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ROBOTS = ["ur10e", "fanuc_m710ic_50"]


@pytest.fixture(scope="module", params=ROBOTS)
def env(request):
    return WorkcellEnv(WorkcellCfg(robot=request.param))


def test_holds_pose_under_gravity(env):
    env.reset(0)
    env.step(500)
    assert np.abs(env.robot.q() - env.robot.observe_q).max() < 2e-3


def test_no_robot_self_or_world_contact_at_observe(env):
    env.reset(0)
    m, d = env.model, env.data
    for c in d.contact[:d.ncon]:
        pair = env.robot.is_robot_geom(np.array([c.geom1, c.geom2]))
        assert not pair.any(), (m.geom(c.geom1).name, m.geom(c.geom2).name)


def test_joint_limits_respected_by_ik(env):
    env.reset(0)
    T = env.robot.top_down(env.meta["tray"]["center"] + [0, 0, 0.2], 0.3)
    r = env.robot.ik(T, q0=env.robot.home_q)
    assert r.success
    assert np.all(r.q >= env.robot.q_lo - 1e-9) and np.all(r.q <= env.robot.q_hi + 1e-9)
    assert np.allclose(env.robot.fk(r.q)[:3, 3], T[:3, 3], atol=1e-3)


def test_top_down_closing_axis(env):
    for yaw in (0.0, 0.7, -2.0):
        T = env.robot.top_down(np.zeros(3), yaw)
        y = T[:3, 1]
        assert abs(abs(np.dot(y, [np.cos(yaw), np.sin(yaw), 0])) - 1) < 1e-9
        assert np.allclose(T[:3, 2], [0, 0, -1])


def test_reset_reproducible(env):
    env.reset(7)
    a = [env.gt_part_pose(i) for i in range(3)]
    env.reset(7)
    b = [env.gt_part_pose(i) for i in range(3)]
    assert all(np.allclose(x, y, atol=1e-9) for x, y in zip(a, b))


def test_oracle_pick_place(env):
    env.reset(3)
    for i in range(3):
        b = ("bin_a", "bin_b")[i % 2]     # bins hold two parts side by side; a third would stack
        r = pick_and_place(env, env.grasp_world(i, env.gt_part_pose(i)), b)
        assert r.reason == "placed", r
        assert env.part_in_bin(i, b)


def test_camera_projection_matches_segmentation(env):
    """GT part centre projected with the believed (== true, no fault) extrinsics lands on that part's mask."""
    env.reset(1)
    f = env.render("fixed", depth=True)
    seg = env.render_segmentation("fixed")
    from aura3d.geometry.camera import PinholeIntrinsics
    K = f.K
    intr = PinholeIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], f.rgb.shape[1], f.rgb.shape[0])
    T_cw = se3.inv_T(f.T_world_cam)
    assert np.allclose(env.gt_true_T_world_cam(), f.T_world_cam, atol=1e-6)
    for i in range(3):
        top = env.gt_part_pose(i)[:3, 3].copy()
        uv, front = intr.project(se3.transform_points(T_cw, top))
        u, v = np.round(uv[0]).astype(int)
        assert front[0] and seg[v, u] == i
