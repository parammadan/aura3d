"""Independently trained action-conditioned latent dynamics model (the fallback world model of master prompt §15).

NOT Meta's JEPA-WM. Same interface idea: z_t = Enc(o_t) with a frozen DINOv2 ViT-S/14 encoder; we train only
  z_{t+1} = z_t + f([z_t, p_t, a_t])          (residual MLP predictor; p = proprio: TCP xyz + jaw width)
  y_hat   = g(z_T)                             (outcome head: P(part lifted at episode end))
with a multi-step open-loop rollout loss over H steps, so errors compound in training as they will in planning.

Latent z = [CLS token, mean patch token] of DINOv2 (768-d), standardised with train-set statistics.
"""
from __future__ import annotations

import torch
from torch import nn


class LatentDynamics(nn.Module):
    def __init__(self, z_dim: int = 768, a_dim: int = 4, p_dim: int = 4, hidden: int = 1024, depth: int = 3):
        super().__init__()
        layers, d = [], z_dim + a_dim + p_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.f = nn.Sequential(*layers, nn.Linear(hidden, z_dim + p_dim))
        self.outcome = nn.Sequential(nn.Linear(z_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.z_dim = z_dim
        nn.init.zeros_(self.f[-1].weight)       # start as the identity map (z_{t+1} = z_t)
        nn.init.zeros_(self.f[-1].bias)

    def step(self, z, p, a):
        d = self.f(torch.cat([z, p, a], -1))
        return z + d[..., : self.z_dim], p + d[..., self.z_dim:]

    def rollout(self, z0, p0, actions):
        """actions (B, H, a_dim) -> predicted latents (B, H, z_dim) and proprio (B, H, p_dim)."""
        zs, ps, z, p = [], [], z0, p0
        for t in range(actions.shape[1]):
            z, p = self.step(z, p, actions[:, t])
            zs.append(z)
            ps.append(p)
        return torch.stack(zs, 1), torch.stack(ps, 1)

    def p_success(self, z):
        return torch.sigmoid(self.outcome(z)).squeeze(-1)
