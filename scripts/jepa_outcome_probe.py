"""EXP-M8-003: can JEPA-WM (WM-B predictor) foresee grasp success at the patch level?

For every traj_v2 episode: encode frame 0, unroll all T = 24 adapted actions with the WM-B predictor, keep the
predicted final visual latent (16x16x384). Train a small attention-pooling probe -> P(success) on train episodes,
early-stop on val AUROC, report val/test AUROC. Control: the same probe on the TRUE final frame's latent.
  pred-probe ~ control  -> the model foresees outcome;   control high, pred low -> it does not;
  control low           -> outcome isn't readable from these latents at all.
Gate (EXP-M8-002 rule): verifier use requires val AUROC >= 0.80 on the PREDICTED latent.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml
from einops import rearrange
from sklearn.metrics import roc_auc_score
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from aura3d.world_models.jepa_adapter import adapt_actions, adapt_frames, adapt_proprio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--wmb", required=True, help="WM-B predictor state dict (jepa_transfer.py output)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
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
model.model.predictor.load_state_dict(torch.load(a.wmb, map_location=dev))
model.model.eval()
mw_mean = np.asarray(dset.proprio_mean, np.float32)
files = {s: sorted(glob.glob(os.path.join(a.data, s, "*.npz"))) for s in ("train", "val", "test")}
c_ours = np.concatenate([np.load(f)["tcp"] for f in files["train"]]).mean(0)


@torch.no_grad()
def episode(f):
    d = np.load(f)
    vis = torch.from_numpy(adapt_frames(d["rgb"][[0, -1]])).to(dev)
    pro = torch.from_numpy(adapt_proprio(d["tcp"][[0, -1]], d["grip_w"][[0, -1]], c_ours, mw_mean)).to(dev)[:, None]
    z = model.encode({"visual": vis, "proprio": pro})
    act = pre.normalize_actions(torch.from_numpy(adapt_actions(d["action"])))
    act = rearrange(act, "t f d -> t 1 (f d)").float().to(dev)
    z0 = TensorDict({"visual": z["visual"][:1].float(), "proprio": z["proprio"][:1].float()}, batch_size=[1])
    pred = model.unroll(z0, act)["visual"][-1, 0]                 # (V, 16, 16, 384)
    return pred.reshape(-1, pred.shape[-1]).half().cpu(), z["visual"][1].reshape(-1, pred.shape[-1]).half().cpu(), \
        float(d["success"])


data = {}
for s, fs in files.items():
    P, Tr, Y = zip(*[episode(f) for f in fs])
    data[s] = (torch.stack(P), torch.stack(Tr), torch.tensor(Y))
    print(s, len(Y), "success", float(torch.tensor(Y).mean()), flush=True)


class Probe(torch.nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = torch.nn.MultiheadAttention(d, 4, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1))

    def forward(self, x):
        o, _ = self.att(self.q.expand(len(x), -1, -1), x, x)
        return self.head(o[:, 0]).squeeze(-1)


def fit(which):
    k = 0 if which == "pred" else 1
    Xtr, Ytr = data["train"][k].float().to(dev), data["train"][2].to(dev)
    pos = Ytr.mean().clamp(0.05, 0.95)
    w = torch.where(Ytr > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    pr = Probe().to(dev)
    opt = torch.optim.AdamW(pr.parameters(), lr=3e-4, weight_decay=0.05)
    best, state = -1, None
    for ep in range(200):
        pr.train()
        perm = torch.randperm(len(Ytr), device=dev)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pr(Xtr[b]), Ytr[b], weight=w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pr.eval()
        with torch.no_grad():
            v = roc_auc_score(data["val"][2].numpy(), pr(data["val"][k].float().to(dev)).cpu().numpy())
        if v > best:
            best, state = v, {n: t.clone() for n, t in pr.state_dict().items()}
    pr.load_state_dict(state)
    with torch.no_grad():
        t = roc_auc_score(data["test"][2].numpy(), pr(data["test"][k].float().to(dev)).cpu().numpy())
    return {"val_auroc": float(best), "test_auroc": float(t)}


res = {"predicted_final_latent": fit("pred"), "true_final_frame_control": fit("true"),
       "gate_val_auroc_min": 0.80}
res["gate_pass"] = res["predicted_final_latent"]["val_auroc"] >= 0.80
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, "probe_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
