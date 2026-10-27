"""Robot-agnostic arm interface: proprioception, FK, damped-least-squares IK, servo commands.

Generalised from parammadan/belief-3d sim/robot/panda.py. Joint names, gains and home poses
come from configs/robots.yaml, so the same class drives the UR10e and the FANUC M-710iC/50.
Only the robot's own model and joint state are used; nothing reads object state. IK runs on
a scratch MjData so solving never perturbs the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aura3d.contracts import RobotState
from aura3d.geometry import se3
from aura3d.simulation.workcell import GRIP_MAX_OPEN

# Tool z points out of the fingers; a top-down grasp has R_WT[:, 2] = -z_W. Fingers close along tool y.
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float      # m
    rot_err: float      # rad
    iters: int


class Manipulator:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: dict, prefix: str = "robot/"):
        self.model, self.data, self.cfg = model, data, cfg
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j) for j in cfg["joints"]]
        if min(jid) < 0:
            raise ValueError(f"joints missing from model: {cfg['joints']}")
        self.jid = np.array(jid)
        self.qpos_adr = model.jnt_qposadr[self.jid]
        self.dof_adr = model.jnt_dofadr[self.jid]
        self.q_lo, self.q_hi = model.jnt_range[self.jid, 0].copy(), model.jnt_range[self.jid, 1].copy()
        # IK may search a narrower box than the joint limits (configuration-branch constraints).
        self.ik_lo = np.maximum(self.q_lo, cfg.get("ik_q_lo", self.q_lo))
        self.ik_hi = np.minimum(self.q_hi, cfg.get("ik_q_hi", self.q_hi))
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j) for j in cfg["joints"]])
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + "ee/gripper")
        self.finger_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                         f"{prefix}ee/finger_{s}")]
                                    for s in ("left", "right")])
        self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "ee/tcp")
        self.home_q = np.array(cfg["home_q"], float)
        self.observe_q = np.array(cfg["observe_q"], float)
        self.T_world_base = se3.make_T(np.eye(3), cfg["base_pos"])
        self._scratch = mujoco.MjData(model)
        self.robot_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}

    # ---------- proprioception ----------
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.finger_adr].sum())

    def tcp_pose(self, data: mujoco.MjData | None = None) -> np.ndarray:
        d = self.data if data is None else data
        return se3.make_T(d.site_xmat[self.tcp].reshape(3, 3), d.site_xpos[self.tcp])

    def state(self) -> RobotState:
        return RobotState(stamp=float(self.data.time), robot_id=self.cfg.get("label", "robot"), q=self.q(),
                          qd=self.data.qvel[self.dof_adr].copy(),
                          T_base_tcp=se3.inv_T(self.T_world_base) @ self.tcp_pose(), gripper_width=self.gripper_width())

    def is_robot_geom(self, geom_ids: np.ndarray) -> np.ndarray:
        return np.isin(self.model.geom_bodyid[geom_ids], list(self.robot_bodies))

    # ---------- reset only ----------
    def set_joint_state(self, q: np.ndarray, gripper_open: bool = True) -> None:
        q = np.clip(q, self.q_lo, self.q_hi)
        self.data.qpos[self.qpos_adr] = q
        self.data.qvel[self.dof_adr] = 0.0
        w = GRIP_MAX_OPEN if gripper_open else 0.0
        self.data.qpos[self.finger_adr] = w
        self.data.ctrl[self.act] = q
        self.data.ctrl[self.grip_act] = w

    # ---------- kinematics ----------
    def fk(self, q: np.ndarray) -> np.ndarray:
        d = self._scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return self.tcp_pose(d)

    def ik(self, T_target: np.ndarray, q0: np.ndarray | None = None, max_iters: int = 300, damping: float = 1e-2,
           pos_tol: float = 1e-3, rot_tol: float = 1e-2) -> IKResult:
        q = np.clip(self.q() if q0 is None else np.asarray(q0, float), self.ik_lo, self.ik_hi)
        d = self._scratch
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        pe = re = np.inf
        for it in range(1, max_iters + 1):
            T = self.fk(q)
            e_p = T_target[:3, 3] - T[:3, 3]
            e_r = T[:3, :3] @ se3.so3_log(T[:3, :3].T @ T_target[:3, :3])
            pe, re = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pe < pos_tol and re < rot_tol:
                return IKResult(q, True, pe, re, it)
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), np.concatenate([e_p, e_r]))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.ik_lo, self.ik_hi)
        return IKResult(q, False, pe, re, max_iters)

    def ik_best(self, targets: list[np.ndarray], seeds: list[np.ndarray] | None = None) -> tuple[int, IKResult]:
        """Successful solution closest to home over all (target, seed) pairs, else lowest error."""
        if seeds is None:
            # Deterministic restarts: DLS from a single seed falls into local minima (UR10e, measured).
            rng = np.random.default_rng(0)
            seeds = [self.home_q, self.q(), self.observe_q] + [
                np.clip(self.home_q + rng.normal(0, 0.6, self.home_q.shape), self.ik_lo, self.ik_hi) for _ in range(4)]
        best = None
        for k, T in enumerate(targets):
            for q0 in seeds:
                r = self.ik(T, q0=q0)
                score = float(np.linalg.norm(r.q - self.home_q)) if r.success else 1e6 + r.pos_err + r.rot_err
                if best is None or score < best[0]:
                    best = (score, k, r)
        return best[1], best[2]

    def top_down(self, p_W: np.ndarray, yaw: float) -> np.ndarray:
        """TCP pose pointing down with the fingers closing along world direction (cos yaw, sin yaw)."""
        return se3.make_T(R_TOP_DOWN @ se3.so3_exp([0.0, 0.0, -(yaw - np.pi / 2)]), p_W)

    # ---------- actuation ----------
    def command(self, q: np.ndarray | None = None, width: float | None = None) -> None:
        if q is not None:
            self.data.ctrl[self.act] = np.clip(q, self.q_lo, self.q_hi)
        if width is not None:
            self.data.ctrl[self.grip_act] = float(np.clip(width / 2, 0.0, GRIP_MAX_OPEN))

    def move_to(self, q_goal: np.ndarray, duration: float, step_fn, settle_tol: float = 2e-3,
                settle_max_s: float = 0.6) -> bool:
        """Minimum-jerk joint-space interpolation of servo targets over `duration` s, then wait
        until every joint is within `settle_tol` rad of the goal (bounded). Returns settled."""
        q0 = self.data.ctrl[self.act].copy()
        n = max(1, round(duration / self.model.opt.timestep))
        for k in range(1, n + 1):
            s = k / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.command(q=q0 + (q_goal - q0) * s)
            step_fn()
        for _ in range(round(settle_max_s / self.model.opt.timestep)):
            if np.abs(self.q() - q_goal).max() < settle_tol:
                return True
            step_fn()
        return bool(np.abs(self.q() - q_goal).max() < settle_tol)
