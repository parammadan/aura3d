"""Closed-loop sorting episode: observe -> localise -> pick one part -> re-observe, until the tray is empty.

The localiser is injected: `oracle_localizer` (B0, privileged, labelled) or any function
CameraFrame -> list[PoseEstimate] (B1+). The controller only ever sees PoseEstimates.
Task: sort by class into bins (SORT_BINS). Attempts are bounded (no infinite retry loops).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aura3d.contracts import PoseEstimate
from aura3d.controllers.pick_place import PickPlaceCfg, pick_and_place
from aura3d.simulation.env import grasp_for

SORT_BINS = {"block": "bin_a", "bracket": "bin_a", "pin": "bin_b"}


@dataclass
class EpisodeResult:
    seed: int
    correct: int                     # parts in their class's bin at the end
    n_parts: int
    attempts: int
    picks: list = field(default_factory=list)
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.correct == self.n_parts


def oracle_localizer(env):
    """PRIVILEGED (B0 only): simulator poses of parts still inside the tray."""
    def loc(_frame):
        out = []
        tc, th = env.meta["tray"]["center"], env.meta["tray"]["half"]
        for i, cls in enumerate(env.meta["part_classes"]):
            T = env.gt_part_pose(i)
            if np.all(np.abs(T[:2, 3] - tc[:2]) < th):
                out.append(PoseEstimate(env.data.time, f"gt{i}:{cls}", "world", T, 1.0, features={"cls": cls}))
        return out
    return loc


def run_episode(env, seed: int, localize, cfg: PickPlaceCfg = PickPlaceCfg(), gate=None,  # noqa: B008
                max_attempts_per_part: int = 2) -> EpisodeResult:
    """gate(estimate, frame) -> (proceed: bool, info) lets reliability layers veto a pick (B3+)."""
    env.reset(seed)
    n = len(env.meta["parts"])
    budget = max_attempts_per_part * n
    picks, attempts = [], 0
    while attempts < budget:
        env.robot.move_to(env.robot.observe_q, cfg.move_s, env.step)
        frame = env.render("fixed")
        ests = localize(frame)
        if not ests:
            break
        est = min(ests, key=lambda e: (e.translation[0], e.translation[1]))   # deterministic order
        if gate is not None:
            ok, info = gate(est, frame)
            if not ok:
                picks.append({"cls": est.features["cls"], "reason": "gated", "info": info})
                attempts += 1
                continue
        cls = est.features["cls"]
        r = pick_and_place(env, grasp_for(cls, est.T, env.meta["table_top"]), SORT_BINS[cls], cfg)
        picks.append({"cls": cls, "reason": r.reason, "cycle_s": r.cycle_s})
        attempts += 1
    correct = sum(env.part_in_bin(i, SORT_BINS[c]) for i, c in enumerate(env.meta["part_classes"]))
    return EpisodeResult(seed, int(correct), n, attempts, picks, "budget" if attempts >= budget else "tray_empty")
