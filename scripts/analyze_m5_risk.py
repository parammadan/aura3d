"""EXP-M5-003 analysis as registered. Writes results/m5/RISK_ANALYSIS.md and risk_results.json."""
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aura3d.uncertainty.risk import FEATURES, GBRisk, risk_coverage

D = Path("datasets/risk_v1")


def load(split):
    parts = [np.load(f) for f in sorted(glob.glob(str(D / f"{split}_*.npz")))]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "e_mm", "cls", "meta")}


tau = json.loads(Path("results/m5/grasp_tolerance_ur10e.json").read_text())["tau_mm"]
tr, cal, te, sh = (load(s) for s in ("train", "cal", "test", "shift"))
tau_of = lambda d: np.array([tau[c] for c in d["cls"]], float)  # noqa: E731
m = GBRisk(alpha=0.1).fit(tr["X"], tr["e_mm"], cal["X"], cal["e_mm"], cal["cls"], tau_of(cal))
iq, ir = FEATURES.index("quality"), FEATURES.index("rms_residual")


def wrong_accept(risk, bad, abstain=0.2):
    keep = np.argsort(risk)[: int(round(len(risk) * (1 - abstain)))]
    return float(bad[keep].mean())


def arms(d):
    return {"a_quality": -d["X"][:, iq], "b_rms": d["X"][:, ir], "c_gb_error": m.predict_error(d["X"]) / tau_of(d),
            "d_gb_p_out": m.p_out(d["X"], tau_of(d))}


out, lines = {"tau_mm": tau, "n": {}}, []
for name, d in (("test", te), ("shift", sh)):
    bad = d["e_mm"] > tau_of(d)
    out["n"][name] = {"rows": int(len(bad)), "out_of_tol": int(bad.sum())}
    res = {}
    for arm, risk in arms(d).items():
        _, _, aurc = risk_coverage(risk, bad)
        res[arm] = {"wrong_accept@20%abst": wrong_accept(risk, bad), "AUROC": float(roc_auc_score(bad, risk)),
                    "AUPRC": float(average_precision_score(bad, risk)), "risk_cov_AUC": aurc}
    # bootstrap over scenes: (c) - (a)
    seeds = d["meta"][:, 0]
    us = np.unique(seeds)
    rng = np.random.default_rng(0)
    A = arms(d)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.nonzero(seeds == s)[0] for s in pick])
        diffs.append(wrong_accept(A["c_gb_error"][idx], bad[idx]) - wrong_accept(A["a_quality"][idx], bad[idx]))
    res["diff_c_minus_a"] = {"mean": float(np.mean(diffs)), "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])]}
    for mond in (False, True):
        ub = m.upper(d["X"], d["cls"], mondrian=mond)
        cov = d["e_mm"] <= ub
        res["conformal_" + ("mondrian" if mond else "marginal")] = {
            "coverage": float(cov.mean()), "mean_bound_mm": float(np.mean(ub)),
            "per_class": {c: float(cov[d["cls"] == c].mean()) for c in np.unique(d["cls"])}}
    out[name] = res
    lines += [f"## {name}  (rows {len(bad)}, out of tolerance {bad.sum()})", "",
              "| arm | wrong-accept @20% abstain | AUROC | AUPRC | risk-coverage AUC |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['wrong_accept@20%abst']:.3f} | {v['AUROC']:.3f} | {v['AUPRC']:.3f} | {v['risk_cov_AUC']:.3f} |"
              for k, v in res.items() if k.startswith(("a_", "b_", "c_", "d_"))]
    dd = res["diff_c_minus_a"]
    lines += ["", f"(c) − (a) wrong-accept: {dd['mean']:+.3f}, 95% CI [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]", ""]
    for k in ("conformal_marginal", "conformal_mondrian"):
        v = res[k]
        lines.append(f"- {k}: coverage {v['coverage']:.3f} (target 0.90), mean bound {v['mean_bound_mm']:.1f} mm, "
                     f"per class {json.dumps({c: round(x, 3) for c, x in v['per_class'].items()})}")
    lines.append("")
Path("results/m5/risk_results.json").write_text(json.dumps(out, indent=1))
hdr = ["# EXP-M5-003 pose-risk results", "", f"τ (mm, from EXP-M5-002): {tau}", ""]
Path("results/m5/RISK_ANALYSIS.md").write_text("\n".join(hdr + lines))
print("\n".join(hdr + lines))
