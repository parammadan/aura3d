"""EXP-M7-001: JEPA-WM (MetaWorld ckpt) on AURA-3D UR10e trajectories. WM-A frozen vs WM-B predictor fine-tune.

Run on the cluster inside the jepa env (sources jepa_runs/env.sh), e.g.
  python scripts/jepa_transfer.py --cfg $R/configs/repro_mw_reach-wall_eps12.yaml --data $D/traj_v2 --out $O [--smoke]

Metric (same protocol as the fallback model, docs/WORLD_MODEL.md): from frame s, unroll H model steps with the
episode's adapted actions; L2 distance to the encoded true frames, compared with (i) copy-z_s and (ii) actions from
another episode. Absolute numbers are in JEPA latent units, so we report the ratio to the copy baseline, which is
comparable across models. WM-B: encoder frozen; predictor + proprio encoder trained on train trajectories with a
multi-step latent L2 loss (visual tokens), AdamW, bf16 autocast, grad clip 1.0, early stopping on val.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from einops import rearrange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from aura3d.world_models.jepa_adapter import adapt_actions, adapt_frames, adapt_proprio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--H", type=int, default=6)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, os.path.join(os.environ["R"], "scripts"))
from jr_stats import make_datasets_from_hub_stats as make_datasets  # noqa: E402
from src.utils.yaml_utils import expand_env_vars  # noqa: E402
from evals.simu_env_planning.eval import init_module  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
args = expand_env_vars(yaml.safe_load(open(a.cfg)))
mk = args["model_kwargs"]
dset, pre = make_datasets(mk["data"], mk["data_aug"], 1, 0)
model = init_module(folder=args["checkpoint_folder"], checkpoint=mk["checkpoint"], module_name=mk["module_name"],
                    model_kwargs=mk["pretrain_kwargs"], wrapper_kwargs=mk["wrapper_kwargs"], cfgs_data=mk["data"],
                    device=dev, action_dim=dset.action_dim, proprio_dim=dset.proprio_dim, preprocessor=pre)
mw_mean = np.asarray(dset.proprio_mean, np.float32)


def files(split):
    f = sorted(glob.glob(os.path.join(a.data, split, "*.npz")))
    return f[:8] if a.smoke else f


c_ours = np.concatenate([np.load(f)["tcp"] for f in files("train")]).mean(0)


@torch.no_grad()
def encode_episode(f):
    d = np.load(f)
    vis = torch.from_numpy(adapt_frames(d["rgb"])).to(dev)
    pro = torch.from_numpy(adapt_proprio(d["tcp"], d["grip_w"], c_ours, mw_mean)).to(dev)[:, None]
    z = model.encode({"visual": vis, "proprio": pro})          # batch over frames
    act = pre.normalize_actions(torch.from_numpy(adapt_actions(d["action"])))          # (T, 5, 4)
    act = rearrange(act, "t f d -> t (f d)").float().to(dev)
    return {"visual": z["visual"].to(torch.bfloat16), "proprio": z["proprio"].to(torch.bfloat16), "act": act,
            "success": bool(d["success"])}


t0 = time.time()
data = {s: [encode_episode(f) for f in files(s)] for s in ("train", "val", "test")}
enc_s = time.time() - t0
print("encoded", {s: len(v) for s, v in data.items()}, round(enc_s, 1), "s", flush=True)


def zslice(e, s):
    # unroll() only returns a value for TensorDict input (a plain dict falls through and returns None upstream)
    from tensordict import TensorDict
    return TensorDict({"visual": e["visual"][s:s + 1].float(), "proprio": e["proprio"][s:s + 1].float()},
                      batch_size=[1])


def rollout(e, s, H, act=None):
    act = e["act"] if act is None else act
    acts = act[s:s + H][:, None]                                # (H, 1, 20)
    out = model.unroll(zslice(e, s), acts)["visual"]            # (H+1 or H, 1, ...)
    return out[-H:, 0]


def evaluate(split, H):
    m = model.eval()
    eps = data[split]
    res = {h: {"model": [], "copy": [], "shuffled": []} for h in range(1, H + 1)}
    with torch.no_grad():
        for k, e in enumerate(eps):
            other = eps[(k + 1) % len(eps)]["act"]
            T = e["act"].shape[0]
            for s in range(0, T - H + 1, 3):
                pr = rollout(e, s, H)
                sh = rollout(e, s, H, act=other)
                for h in range(1, H + 1):
                    tgt = e["visual"][s + h].float()
                    res[h]["model"].append(float((pr[h - 1] - tgt).norm()))
                    res[h]["shuffled"].append(float((sh[h - 1] - tgt).norm()))
                    res[h]["copy"].append(float((e["visual"][s].float() - tgt).norm()))
    out = {}
    for h, r in res.items():
        mm, cc, ss = (float(np.mean(r[k])) for k in ("model", "copy", "shuffled"))
        out[h] = {"model": mm, "copy": cc, "shuffled": ss, "model_over_copy": mm / cc, "shuffled_over_copy": ss / cc}
    return out


results = {"H": a.H, "smoke": a.smoke, "encode_s": enc_s, "adapter": "aura3d.world_models.jepa_adapter",
           "n": {s: len(v) for s, v in data.items()}, "gpu": torch.cuda.get_device_name(0)}
results["WM-A_frozen"] = {"val": evaluate("val", a.H), "test": evaluate("test", a.H)}
print("WM-A test", json.dumps({h: round(v["model_over_copy"], 3) for h, v in results["WM-A_frozen"]["test"].items()}),
      flush=True)

# ---- WM-B: fine-tune predictor (+ proprio encoder); encoder frozen ----
for p in model.model.parameters():
    p.requires_grad = False
train_params = list(model.model.predictor.parameters())
if isinstance(model.model.proprio_encoder, torch.nn.Module):
    train_params += list(model.model.proprio_encoder.parameters())
for p in train_params:
    p.requires_grad = True
results["trainable_params"] = sum(p.numel() for p in train_params)
opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.01)
best, best_state, log = np.inf, None, []
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
for ep in range(1 if a.smoke else a.epochs):
    model.model.train()
    order = np.random.default_rng(ep).permutation(len(data["train"]))
    losses = []
    for k in order:
        e = data["train"][k]
        T = e["act"].shape[0]
        s = int(np.random.default_rng([ep, int(k)]).integers(0, T - a.H + 1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = rollout(e, s, a.H)
            tgt = e["visual"][s + 1:s + 1 + a.H].float()
            loss = ((pr.float() - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        losses.append(float(loss))
    v = evaluate("val", a.H)[a.H]["model_over_copy"]
    log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_ratio_H": v})
    print(json.dumps(log[-1]), flush=True)
    if v < best:
        best = v
        best_state = {k: t.detach().clone() for k, t in model.model.predictor.state_dict().items()}
model.model.predictor.load_state_dict(best_state)
results["WM-B_log"] = log
results["train_s"] = time.time() - t1
results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
results["WM-B_predictor_ft"] = {"val": evaluate("val", a.H), "test": evaluate("test", a.H)}
torch.save(best_state, os.path.join(a.out, "wmB_predictor.pt"))
json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("WM-B test", json.dumps({h: round(v["model_over_copy"], 3) for h, v in results["WM-B_predictor_ft"]["test"].items()}))
