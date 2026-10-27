"""EXP-M5-001 analysis (as registered): per cell Wilson CIs, paired exact McNemar B3cal vs B1.

Also reports what the registry calls secondary: alarms at zero drift, deferrals (episode ended with no
pick because recalibration was refused) and wrong actions (executed picks that did not place a part).
Writes results/m5/ANALYSIS.md.
"""
import json
from pathlib import Path

from aura3d.evaluation.stats import mcnemar_exact, wilson

R = Path("results/m5")
cells = [(0, 0), (0.25, 0), (0.5, 0), (1, 0), (2, 0), (0, 5), (0, 10), (1, 10), (3, 20)]


def load(b, r, t):
    p = R / f"{b}_ur10e_nominal_r{r:g}_t{t:g}.json"
    return json.loads(p.read_text()) if p.exists() else None


def wrong(e):
    return sum(p["reason"] not in ("placed", "gated") for p in e["picks"])


rows = ["| drift (deg, mm) | B1 episodes | B3cal episodes | Δ (paired) | McNemar p | B1 wrong picks | B3cal wrong picks | B3cal deferred eps | B3cal alarm eps |",
        "|---|---|---|---|---|---|---|---|---|"]
for r, t in cells:
    a, b = load("B1", r, t), load("B3cal", r, t)
    if a is None or b is None:
        rows.append(f"| {r}, {t} | (missing) | | | | | | | |")
        continue
    ea = {e["seed"]: e for e in a["episodes"]}
    eb = {e["seed"]: e for e in b["episodes"]}
    seeds = sorted(set(ea) & set(eb))
    ka, kb = sum(ea[s]["success"] for s in seeds), sum(eb[s]["success"] for s in seeds)
    n = len(seeds)
    bc = sum(ea[s]["success"] and not eb[s]["success"] for s in seeds)
    cb = sum(eb[s]["success"] and not ea[s]["success"] for s in seeds)
    deferred = sum(any(c.get("recal") not in (None, "ok") for c in eb[s]["calib"]) for s in seeds)
    alarms = sum(any(c["alarm"] for c in eb[s]["calib"]) for s in seeds)
    wa, wb = sum(wrong(ea[s]) for s in seeds), sum(wrong(eb[s]) for s in seeds)
    la, ha = wilson(ka, n)
    lb, hb = wilson(kb, n)
    rows.append(f"| {r}, {t} | {ka}/{n} [{la:.2f}, {ha:.2f}] | {kb}/{n} [{lb:.2f}, {hb:.2f}] | {(kb - ka) / n:+.2f} "
                f"| {mcnemar_exact(bc, cb):.2g} | {wa} | {wb} | {deferred} | {alarms} |")
mon = load("B3cal", 0, 0)
hdr = ["# EXP-M5-001 results: camera drift dose-response (UR10e, nominal noise, seeds 2000–2049)", "",
       f"Monitor (commissioned on dev seeds 0–19): μ0 = {mon['summary']['monitor']['mu0']:.3f} mm, "
       f"σ0 = {mon['summary']['monitor']['sigma0']:.3f} mm, threshold = {mon['summary']['monitor']['thr']:.3f} mm."
       if mon else "", "",
       "Episode success = all 3 parts in their class bin. Wrong picks = executed picks that did not place a part. "
       "Deferred = recalibration refused (validation failed), so the cell stopped and asked for an operator; this counts "
       "as episode failure but is not a wrong action.", ""]
(R / "ANALYSIS.md").write_text("\n".join(hdr + rows) + "\n")
print("\n".join(rows))
