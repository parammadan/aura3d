"""State machine terminates within budget and logs every transition (no infinite recovery loops)."""
import numpy as np

from aura3d.contracts import PoseEstimate
from aura3d.recovery.state_machine import AuraRunner, Budget
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg


class AlwaysRisky:
    def p_out(self, X, tau):
        return np.ones(len(X))


def _phantom(_frame):
    T = np.eye(4)
    T[:3, 3] = [0.75, -0.15, 0.015]
    return [PoseEstimate(0.0, "p", "world", T, 1.0, features={"cls": "block", "rms_residual": 0.001, "n_points": 100,
                                                                "pixels": 100, "spread_xy": 0.01,
                                                                "valid_depth_frac": 1.0})]


def test_risky_estimates_defer_after_bounded_reobserve():
    env = WorkcellEnv(WorkcellCfg(robot="ur10e"))
    env.reset(0)
    log = AuraRunner(env, _phantom, risk=(AlwaysRisky(), {"block": 15.0}), budget=Budget(max_reobserve=2)).run()
    assert log.outcome == "DEFER"
    assert sum("reobserve" in t[2] for t in log.transitions) == 2
    assert not log.picks


def test_failed_part_is_not_retried_forever():
    env = WorkcellEnv(WorkcellCfg(robot="ur10e"))
    env.reset(0)
    log = AuraRunner(env, _phantom, budget=Budget(max_fail_per_part=1)).run()   # phantom: every pick misses
    assert log.outcome == "COMPLETE"            # the phantom is excluded after one failure
    assert len(log.picks) == 1
