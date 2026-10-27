"""AURA-3D recovery state machine (master prompt §17): the full reliability loop around the pick controller.

  OBSERVE -> CHECK_CALIB -(alarm)-> RECALIBRATE -(refused)-> DEFER
                |                        |
                v                        v (accepted: new extrinsics, re-observe)
            LOCALIZE -> EVALUATE_RISK -(p_out > thr)-> REOBSERVE (bounded) -> ... -> skip part / DEFER
                               | (ok)
                               v
                          EXECUTE -> VERIFY -(fail)-> mark part, try another (bounded)
                               | (placed)
                               v
                           OBSERVE ... COMPLETE when nothing is left
Budgets: max_steps transitions, max_recal recalibrations, max_reobserve per target, max_fail per part.
Every transition is logged as (state, next_state, reason). No state can loop without consuming budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aura3d.calibration import fixture as fx
from aura3d.controllers.episode import SORT_BINS
from aura3d.controllers.pick_place import PickPlaceCfg, pick_and_place
from aura3d.simulation.env import grasp_for
from aura3d.uncertainty.risk import featurize


@dataclass
class Budget:
    max_steps: int = 80
    max_recal: int = 2
    max_reobserve: int = 2
    max_fail_per_part: int = 1


@dataclass
class RunLog:
    transitions: list = field(default_factory=list)
    picks: list = field(default_factory=list)
    outcome: str = ""

    def t(self, a, b, why):
        self.transitions.append((a, b, why))


class AuraRunner:
    """monitor: DriftMonitor or None; risk: (GBRisk, tau dict) or None; localize: frame -> estimates."""

    def __init__(self, env, localize, monitor=None, risk=None, p_out_thr: float = 0.5, budget: Budget = Budget(),  # noqa: B008
                 bin_slots: bool = False):
        self.env, self.localize, self.monitor, self.risk = env, localize, monitor, risk
        self.thr, self.budget, self.bin_slots = p_out_thr, budget, bin_slots

    def run(self, cfg: PickPlaceCfg = PickPlaceCfg()) -> RunLog:  # noqa: B008
        env, B, log = self.env, self.budget, RunLog()
        slots = [(-0.04, -0.04), (0.04, 0.04), (-0.04, 0.04), (0.04, -0.04)]
        used, failed, reobs, recal_n = {}, [], 0, 0
        state, steps = "OBSERVE", 0
        frame = ests = est = None
        r_cal = 0.0
        while steps < B.max_steps:
            steps += 1
            if state == "OBSERVE":
                env.robot.move_to(env.robot.observe_q, cfg.move_s, env.step)
                frame = env.render("fixed")
                nxt = "CHECK_CALIB" if self.monitor is not None else "LOCALIZE"
                log.t(state, nxt, "frame")
                state = nxt
            elif state == "CHECK_CALIB":
                alarm, r_cal = self.monitor.check(frame, env.meta)
                nxt = "RECALIBRATE" if alarm else "LOCALIZE"
                log.t(state, nxt, f"residual_mm={r_cal:.2f}")
                state = nxt
            elif state == "RECALIBRATE":
                if recal_n >= B.max_recal:
                    log.t(state, "DEFER", "recal_budget")
                    state = "DEFER"
                    continue
                recal_n += 1
                rc = fx.recalibrate(frame, env.meta)
                if rc.status != "ok":
                    log.t(state, "DEFER", f"recal_{rc.status}")
                    state = "DEFER"
                    continue
                env.believed_T_world_cam["fixed"] = rc.T_world_cam
                log.t(state, "OBSERVE", "recal_ok")
                state = "OBSERVE"
            elif state == "LOCALIZE":
                ests = [e for e in self.localize(frame)
                        if sum(np.linalg.norm(e.translation[:2] - q) < 0.02 for q in failed) < B.max_fail_per_part]
                if not ests:
                    log.t(state, "COMPLETE", "nothing_left")
                    state = "COMPLETE"
                    continue
                log.t(state, "EVALUATE_RISK", f"n={len(ests)}")
                state = "EVALUATE_RISK"
            elif state == "EVALUATE_RISK":
                if self.risk is None:
                    est = min(ests, key=lambda e: (e.translation[0], e.translation[1]))
                    log.t(state, "EXECUTE", "no_risk_model")
                    state = "EXECUTE"
                    continue
                model, tau = self.risk
                X = np.stack([featurize(e, r_cal) for e in ests])
                taus = np.array([tau[e.features["cls"]] for e in ests])
                p = model.p_out(X, taus)
                k = int(np.argmin(p))
                if p[k] <= self.thr:
                    est = ests[k]
                    log.t(state, "EXECUTE", f"p_out={p[k]:.2f}")
                    state = "EXECUTE"
                elif reobs < B.max_reobserve:
                    reobs += 1
                    log.t(state, "OBSERVE", f"reobserve p_out_min={p[k]:.2f}")
                    state = "OBSERVE"
                else:
                    log.t(state, "DEFER", f"risk p_out_min={p[k]:.2f}")
                    state = "DEFER"
            elif state == "EXECUTE":
                cls = est.features["cls"]
                b = SORT_BINS[cls]
                off = slots[used.get(b, 0) % 4] if self.bin_slots else (0.0, 0.0)
                r = pick_and_place(env, grasp_for(cls, est.T, env.meta["table_top"]), b, cfg, place_offset=off)
                log.picks.append({"cls": cls, "reason": r.reason})
                if r.reason == "placed":
                    used[b] = used.get(b, 0) + 1
                    reobs = 0
                    log.t(state, "OBSERVE", "placed")
                else:
                    failed.append(est.translation[:2].copy())
                    log.t(state, "OBSERVE", f"verify_fail:{r.reason}")
                state = "OBSERVE"
            elif state in ("COMPLETE", "DEFER"):
                break
        log.outcome = state if state in ("COMPLETE", "DEFER") else "BUDGET"
        return log
