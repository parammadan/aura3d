"""Workcell runtime: reproducible resets, physics stepping, RGB-D rendering and privileged ground truth.

Privileged accessors are prefixed `gt_` and may only be called by data generation, oracle
baselines, evaluation and labelled debug tools (master prompt §7). Perception and planning
receive CameraFrame / RobotState only.
"""
from __future__ import annotations

import mujoco
import numpy as np

from aura3d.contracts import CameraFrame
from aura3d.geometry import se3
from aura3d.robots.manipulator import Manipulator
from aura3d.simulation.workcell import CATALOG, WorkcellCfg, build_workcell

# Per-class grasp in the part frame: offset (m), closing axis, TCP height above the table (m).
GRASPS = {
    "block": {"offset": (0.0, 0.0), "axis": (0.0, 1.0), "tcp_z": 0.012, "width": 0.040},
    "pin": {"offset": (0.0, 0.0), "axis": (0.0, 1.0), "tcp_z": 0.020, "width": 0.032},
    "bracket": {"offset": (-0.025, 0.0), "axis": (0.0, 1.0), "tcp_z": 0.012, "width": 0.030},
}
PART_HALF_HEIGHT = {"block": 0.015, "pin": 0.025, "bracket": 0.015}


class WorkcellEnv:
    def __init__(self, cfg: WorkcellCfg | None = None):
        self.cfg = cfg or WorkcellCfg()
        self.model, self.meta = build_workcell(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.robot = Manipulator(self.model, self.data, self.meta["robot_cfg"])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.part_bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.meta["parts"]]
        self.part_qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.part_bodies]
        self.cam_ids = {k: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam/{k}")
                        for k in self.meta["cameras"]}
        # Extrinsics the system believes; drift faults change the true camera, not this.
        self.believed_T_world_cam = {k: v["T_world_cam"].copy() for k, v in self.meta["cameras"].items()}

    # ---------- reset ----------
    def reset(self, seed: int, min_sep: float = 0.085) -> dict:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.robot.set_joint_state(self.robot.observe_q)
        tray, top = self.meta["tray"], self.meta["table_top"]
        placed = []
        for qadr, cls in zip(self.part_qadr, self.meta["part_classes"]):
            for _ in range(1000):
                xy = tray["center"][:2] + rng.uniform(-1, 1, 2) * (tray["half"] - 0.055)
                if all(np.linalg.norm(xy - p) > min_sep for p in placed):
                    break
            else:
                raise RuntimeError("could not place parts without overlap")
            placed.append(xy)
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[qadr:qadr + 3] = [xy[0], xy[1], top + PART_HALF_HEIGHT[cls] + 0.001]
            self.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(250):            # settle parts on the table
            self.step()
        return {"seed": seed}

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, n)

    # ---------- sensing ----------
    def render(self, cam: str = "fixed", depth: bool = True) -> CameraFrame:
        c = self.meta["cameras"][cam]
        key = (c["height"], c["width"])
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, c["height"], c["width"])
        r = self._renderers[key]
        r.update_scene(self.data, camera=self.cam_ids[cam])
        rgb = r.render().copy()
        d = None
        if depth:
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=self.cam_ids[cam])
            d = r.render().astype(np.float32)     # MuJoCo returns z-depth in metres
            r.disable_depth_rendering()
        return CameraFrame(stamp=float(self.data.time), frame_id=f"camera/{cam}", rgb=rgb, depth=d, K=c["K"],
                           T_world_cam=self.believed_T_world_cam[cam])

    def render_segmentation(self, cam: str = "fixed") -> np.ndarray:
        """PRIVILEGED: (H, W) part index per pixel, -1 for background. Labels only."""
        c = self.meta["cameras"][cam]
        r = self._renderers.setdefault((c["height"], c["width"]), mujoco.Renderer(self.model, c["height"], c["width"]))
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.cam_ids[cam])
        seg = r.render()
        r.disable_segmentation_rendering()
        geom_id = seg[..., 0]
        out = np.full(geom_id.shape, -1, np.int32)
        valid = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (geom_id >= 0)
        body = np.where(valid, self.model.geom_bodyid[np.clip(geom_id, 0, None)], -1)
        for i, b in enumerate(self.part_bodies):
            out[body == b] = i
        return out

    # ---------- privileged ground truth ----------
    def gt_part_pose(self, i: int) -> np.ndarray:
        b = self.part_bodies[i]
        return se3.make_T(self.data.xmat[b].reshape(3, 3), self.data.xpos[b])

    def gt_true_T_world_cam(self, cam: str = "fixed") -> np.ndarray:
        cid = self.cam_ids[cam]
        R_mj = self.data.cam_xmat[cid].reshape(3, 3)
        return se3.make_T(R_mj @ np.diag([1.0, -1.0, -1.0]), self.data.cam_xpos[cid])

    def set_true_camera_pose(self, T_world_cam: np.ndarray, cam: str = "fixed") -> None:
        """Fault injection: physically move the camera. Believed extrinsics stay unchanged."""
        cid = self.cam_ids[cam]
        self.model.cam_pos[cid] = T_world_cam[:3, 3]
        self.model.cam_quat[cid] = se3.rot_to_quat(T_world_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0]))
        mujoco.mj_forward(self.model, self.data)

    def part_in_bin(self, i: int, bin_name: str) -> bool:
        p = self.data.xpos[self.part_bodies[i]]
        b = self.meta["bins"][bin_name]
        inside = np.all(np.abs(p[:2] - b["center"][:2]) < b["half"] - 0.005)
        return bool(inside and p[2] < self.meta["table_top"] + 0.06)

    def grasp_world(self, i: int, T_world_part: np.ndarray) -> dict:
        """Grasp target for part i given ITS POSE (estimated or oracle) — the pose source decides privilege."""
        g = GRASPS[self.meta["part_classes"][i]]
        R = T_world_part[:3, :3]
        p = T_world_part[:3, 3] + R @ np.array([*g["offset"], 0.0])
        p[2] = self.meta["table_top"] + g["tcp_z"]
        axis = R @ np.array([*g["axis"], 0.0])
        return {"p": p, "yaw": float(np.arctan2(axis[1], axis[0])), "width": g["width"]}


__all__ = ["WorkcellEnv", "GRASPS", "CATALOG"]
