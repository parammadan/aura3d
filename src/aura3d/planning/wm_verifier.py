"""World-model grasp verification (the predictive-planning component, M8).

Given the current 'wm' camera frame, proprio, and K candidate grasp points (from perception plus offsets within
its risk bound), build each candidate's scripted action sequence, roll it out in the latent dynamics model and
score P(success) with the outcome head. Select argmax; return DEFER when the best score is below p_min.
This uses the model only for ranking/abstaining among geometrically valid candidates. Every executed motion
is still the deterministic controller's, so the world model never commands the arm directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aura3d.controllers.delta_policy import scripted_actions
from aura3d.world_model_training.train_latent import MEAN, STD
from aura3d.world_models.latent_dynamics import LatentDynamics


@dataclass
class Choice:
    index: int
    p_success: np.ndarray
    defer: bool


class WMVerifier:
    def __init__(self, ckpt: str, device: str = "cpu"):
        st = torch.load(ckpt, map_location=device)
        self.stats = st["stats"]
        self.model = LatentDynamics().to(device).eval()
        self.model.load_state_dict(st["model"])
        self.enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        self.dev = device

    @torch.no_grad()
    def encode(self, rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255)
        o = self.enc.forward_features(((x - MEAN) / STD).to(self.dev))
        z = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], -1)
        return (z - self.stats["zm"]) / self.stats["zs"]

    @torch.no_grad()
    def score(self, rgb: np.ndarray, tcp: np.ndarray, jaw: float, goals: list[np.ndarray]) -> np.ndarray:
        z0 = self.encode(rgb)
        p0 = (torch.tensor([*tcp, jaw]).float()[None] - self.stats["pm"]) / self.stats["ps"]
        acts = torch.from_numpy(np.stack([scripted_actions(tcp, g) for g in goals])) / self.stats["as"]
        K = len(goals)
        zs, _ = self.model.rollout(z0.expand(K, -1), p0.expand(K, -1), acts)
        return self.model.p_success(zs[:, -1]).numpy()

    def choose(self, rgb, tcp, jaw, goals, p_min: float = 0.5) -> Choice:
        """Ties (e.g. a saturated outcome head) break toward the candidate nearest the perception estimate,
        which callers put at goals[len//2]."""
        p = self.score(rgb, tcp, jaw, goals)
        c = np.asarray(goals[len(goals) // 2])
        dist = np.array([np.linalg.norm(np.asarray(g) - c) for g in goals])
        k = int(np.argmax(p - 1e-3 * dist))
        return Choice(k, p, bool(p[k] < p_min))
