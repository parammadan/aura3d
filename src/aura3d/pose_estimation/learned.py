"""B2 perception: fine-tuned Mask R-CNN instances + the B1 depth/SDF pose fit.

Only the segmentation and class decision change relative to B1 (classical.py). Each instance mask with
score >= min_score selects depth points; the same tray/height gates and `fit_planar_pose` are applied, so any
difference between B1 and B2 is attributable to learned instance segmentation and classification.
"""
from __future__ import annotations

import numpy as np
import torch

from aura3d.contracts import CameraFrame, PoseEstimate
from aura3d.geometry import se3
from aura3d.geometry.camera import PinholeIntrinsics
from aura3d.pose_estimation.classical import ClassicalCfg, fit_planar_pose

CLASSES = ["block", "pin", "bracket"]


class MaskRCNNLocalizer:
    def __init__(self, ckpt: str, device: str = "cuda", min_score: float = 0.5, mask_thr: float = 0.5):
        from aura3d.vision_training.train_maskrcnn import build_model
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_model("FT-A")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.dev)["model"])
        self.model.to(self.dev).eval()
        self.min_score, self.mask_thr = min_score, mask_thr

    @torch.no_grad()
    def instances(self, rgb: np.ndarray):
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.dev)
        o = self.model([x])[0]
        keep = o["scores"] >= self.min_score
        return ((o["masks"][keep, 0] > self.mask_thr).cpu().numpy(), o["labels"][keep].cpu().numpy(),
                o["scores"][keep].cpu().numpy())

    def __call__(self, frame: CameraFrame, layout: dict, cfg: ClassicalCfg = ClassicalCfg()) -> list[PoseEstimate]:  # noqa: B008
        if frame.depth is None:
            return []
        masks, labels, scores = self.instances(frame.rgb)
        h, w = frame.depth.shape
        K = frame.K
        intr = PinholeIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h)
        top = layout["table_top"]
        tc, th = layout["tray"]["center"], layout["tray"]["half"] - cfg.tray_margin
        out = []
        for k, (m, lab, sc) in enumerate(zip(masks, labels, scores)):
            d = frame.depth
            ok = m & np.isfinite(d) & (d > 0)
            v, u = np.nonzero(ok)
            if len(v) < cfg.min_pixels:
                continue
            P = se3.transform_points(frame.T_world_cam, intr.backproject(u, v, d[v, u]))
            sel = ((P[:, 2] > top + cfg.min_height) & (P[:, 2] < top + cfg.max_height)
                   & np.all(np.abs(P[:, :2] - tc[:2]) < th, axis=1))
            P = P[sel]
            if len(P) < cfg.min_pixels:
                continue
            cls = CLASSES[int(lab) - 1]
            T, feats = fit_planar_pose(cls, P, top, cfg.huber)
            feats.update({"pixels": int(m.sum()), "valid_depth_frac": float(ok.sum() / max(1, m.sum())),
                          "mask_score": float(sc), "bbox": [int(u.min()), int(v.min()), int(u.max()), int(v.max())]})
            out.append(PoseEstimate(stamp=frame.stamp, object_id=f"mrcnn{k}:{cls}", frame_id="world", T=T,
                                    quality=float(sc * np.exp(-feats["rms_residual"] / 0.002)), features={"cls": cls, **feats}))
        return out
