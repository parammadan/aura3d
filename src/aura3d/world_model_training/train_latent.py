"""Train the fallback latent dynamics model on recorded trajectories (GPU).

Step 1 (once): encode every frame with frozen DINOv2 ViT-S/14 at 224x224 (ImageNet normalisation) and cache
           features to <data>/<split>_dinov2.pt.
Step 2: train LatentDynamics with an H-step open-loop rollout loss (MSE on standardised latents + proprio) plus
        BCE on the outcome head at the final step. Report held-out prediction error per horizon, an
        action-shuffle control (same z0, actions from another episode) and outcome AUROC.
  python -m aura3d.world_model_training.train_latent --data datasets/traj_v1 --out runs/wm_latent
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aura3d.world_models.latent_dynamics import LatentDynamics

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode_split(data: Path, split: str, dev) -> dict:
    cache = data / f"{split}_dinov2.pt"
    if cache.exists():
        return torch.load(cache)
    enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    Z, P, A, Y, S = [], [], [], [], []
    for f in sorted(glob.glob(str(data / split / "*.npz"))):
        d = np.load(f)
        x = torch.from_numpy(d["rgb"]).permute(0, 3, 1, 2).float().div(255)
        x = ((x - MEAN) / STD).to(dev)
        out = enc.forward_features(x)
        z = torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], -1).cpu()
        Z.append(z)
        P.append(torch.from_numpy(np.concatenate([d["tcp"], d["grip_w"][:, None]], 1)).float())
        A.append(torch.from_numpy(d["action"]).float())
        Y.append(float(d["success"]))
        S.append(int(d["seed"]))
    res = {"z": torch.stack(Z), "p": torch.stack(P), "a": torch.stack(A), "y": torch.tensor(Y), "seed": torch.tensor(S)}
    torch.save(res, cache)
    return res


def windows(d, H, stats):
    z = (d["z"] - stats["zm"]) / stats["zs"]
    p = (d["p"] - stats["pm"]) / stats["ps"]
    a = d["a"] / stats["as"]
    return z, p, a


def eval_split(model, d, stats, H, dev):
    z, p, a = (t.to(dev) for t in windows(d, H, stats))
    model.eval()
    with torch.no_grad():
        T = a.shape[1]
        zs, ps = model.rollout(z[:, 0], p[:, 0], a)
        err = ((zs - z[:, 1:]) ** 2).mean(-1)                      # (N, T) per step
        copy = ((z[:, :1] - z[:, 1:]) ** 2).mean(-1)               # "nothing moves" baseline
        perm = torch.roll(torch.arange(len(a)), 1)
        zsh, _ = model.rollout(z[:, 0], p[:, 0], a[perm])          # actions from a different episode
        err_sh = ((zsh - z[:, 1:]) ** 2).mean(-1)
        # L2-norm ratios vs copy-z0, the scale-free metric shared with scripts/jepa_transfer.py
        n_model = (zs - z[:, 1:]).norm(dim=-1)
        n_copy = (z[:, :1] - z[:, 1:]).norm(dim=-1)
        n_sh = (zsh - z[:, 1:]).norm(dim=-1)
        pr = model.p_success(zs[:, -1]).cpu().numpy()
    y = d["y"].numpy()
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(y, pr)) if 0 < y.mean() < 1 else float("nan")
    hs = [1, 2, 4, 8, T]
    return {"mse_by_h": {h: float(err[:, h - 1].mean()) for h in hs},
            "copy_baseline_by_h": {h: float(copy[:, h - 1].mean()) for h in hs},
            "shuffled_actions_by_h": {h: float(err_sh[:, h - 1].mean()) for h in hs},
            "model_over_copy_by_h": {h: float(n_model[:, h - 1].mean() / n_copy[:, h - 1].mean()) for h in range(1, T + 1)},
            "shuffled_over_copy_by_h": {h: float(n_sh[:, h - 1].mean() / n_copy[:, h - 1].mean()) for h in range(1, T + 1)},
            "outcome_auroc": auroc, "success_rate": float(y.mean()), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outcome-weight", type=float, default=0.1)
    ap.add_argument("--balanced", action="store_true", help="class-balanced BCE for the outcome head")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va, te = (encode_split(data, s, dev) for s in ("train", "val", "test"))
    enc_s = time.time() - t0
    stats = {"zm": tr["z"].mean((0, 1)), "zs": tr["z"].std((0, 1)) + 1e-6, "pm": tr["p"].mean((0, 1)),
             "ps": tr["p"].std((0, 1)) + 1e-6, "as": tr["a"].abs().amax((0, 1)) + 1e-6}
    model = LatentDynamics().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    z, p, act = (t.to(dev) for t in windows(tr, a.H, stats))
    y = tr["y"].to(dev)
    N, T = act.shape[:2]
    log, best = [], (np.inf, None)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, 64):
            b = perm[i:i + 64]
            s = torch.randint(0, T - a.H + 1, (1,)).item()
            zs, ps = model.rollout(z[b, s], p[b, s], act[b, s:s + a.H])
            loss = F.mse_loss(zs, z[b, s + 1:s + 1 + a.H]) + F.mse_loss(ps, p[b, s + 1:s + 1 + a.H])
            zT, _ = model.rollout(z[b, 0], p[b, 0], act[b])
            w = None
            if a.balanced:
                pos = y.mean().clamp(0.05, 0.95)
                w = torch.where(y[b] > 0.5, 0.5 / pos, 0.5 / (1 - pos))
            loss = loss + a.outcome_weight * F.binary_cross_entropy(
                model.p_success(zT[:, -1]).clamp(1e-5, 1 - 1e-5), y[b], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 25 == 0 or ep == a.epochs - 1:
            v = eval_split(model, va, stats, a.H, dev)
            log.append({"epoch": ep, "loss": float(loss), "val_mse_h8": v["mse_by_h"][8]})
            print(json.dumps(log[-1]), flush=True)
            if v["mse_by_h"][8] < best[0]:
                best = (v["mse_by_h"][8], {k: t.detach().clone() for k, t in model.state_dict().items()})
    model.load_state_dict(best[1])
    res = {"model": "LatentDynamics (ours, NOT JEPA-WM)", "encoder": "dinov2_vits14 frozen", "args": vars(a),
           "config_hash": hashlib.sha256(json.dumps(vars(a), sort_keys=True).encode()).hexdigest()[:12],
           "encode_s": enc_s, "train_s": time.time() - t0 - enc_s, "log": log,
           "val": eval_split(model, va, stats, a.H, dev), "test": eval_split(model, te, stats, a.H, dev),
           "params": sum(p.numel() for p in model.parameters()),
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    torch.save({"model": model.state_dict(), "stats": stats, "args": vars(a)}, out / "best.pt")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("val", "test")}, indent=1))


if __name__ == "__main__":
    main()
