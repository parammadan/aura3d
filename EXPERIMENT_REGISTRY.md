# Experiment registry

Entries are written **before** the test run. Thresholds, seeds and metrics in an entry are frozen once it
is committed; changes after seeing results get a new entry. Dev seeds 0–999; test seeds ≥ 1000, never
used for tuning.

## EXP-M2-001: B0 oracle controller (done 2026-10-27)
- Question: does the deterministic controller complete the task when given exact poses?
- Seeds 1000–1099, 3 parts each, UR10e and FANUC M-710iC/50. Metric: parts in bin (Wilson 95 %).
- Result: `results/m2/b0_*.json`.

## EXP-M3-001: B0 vs B1 closed-loop sorting, sensor noise only (done 2026-10-27)
- Seeds 1000–1099; noise presets nominal and degraded; both robots. Metric: episode success.
- Result: `results/m3/`.

## EXP-M5-001: camera drift dose-response, B1 vs B3cal (registered 2026-10-27, before running)
- **Question (H2, partial):** does fixture-residual monitoring with validated recalibration keep the cell
  working after a camera bump, and at what false-alarm cost?
- **Arms:** B1 (classical perception, no monitoring) and B3cal (B1 + DriftMonitor + fixture recalibration,
  leave-one-sphere-out validation; defers on failed validation).
- **Fault:** sudden camera bump before the episode, applied in the camera frame, random direction per seed.
  Grid (rot °, trans mm): (0,0) (0.25,0) (0.5,0) (1,0) (2,0) (0,5) (0,10) (1,10) (3,20).
- **Fixed settings:** UR10e, nominal noise, reference spheres on. Monitor commissioned on dev seeds 0–19, alarm at
  μ0 + 6σ0. Recal acceptance: held-out sphere error ≤ 2 mm on ≥ 2 spheres.
- **Seeds:** 2000–2049 (50 episodes per cell, paired across arms: same layout and same drift direction).
- **Primary metric:** episode success (all 3 parts in their bins). **Secondary:** alarm rate at (0,0) (false
  alarms), deferral rate, parts correct.
- **Analysis:** per cell, Wilson 95 % CIs; paired difference B3cal − B1 with exact McNemar test on discordant episodes.
  No threshold changes after this entry.

## EXP-M5-002: per-class grasp tolerance (dev, registered 2026-10-27)
- Oracle grasp offset by δ ∈ {0,2,4,6,8,10,12,15} mm in a random xy direction; 40 dev seeds per class (0–39).
- τ_cls = largest δ with success ≥ 95 %. τ is frozen before EXP-M5-003 touches test data.

## EXP-M5-003: pose-risk estimator vs confidence threshold (H1) (registered 2026-10-27, before test data)
- **Data:** `scripts/make_risk_dataset.py`. Train dev 0–599, calibration dev 600–899, test 3000–3299 (same
  distribution), shift 3300–3499 (degraded noise, larger bumps, more clutter). One row per detection.
- **Label:** grasp-point xy error e (mm, capped at 50); out of tolerance := e > τ_cls (EXP-M5-002).
- **Arms:** (a) estimator quality score (risk = −quality); (b) registration RMS alone; (c) GBRisk predicted error;
  (d) GBRisk + isotonic P(out).
- **Primary:** wrong-accept rate (accepted rows that are out of tolerance) at a matched 20 % abstention, on test.
  **Secondary:** AUROC, AUPRC, risk-coverage AUC; split-conformal (α = 0.1) empirical coverage and mean bound for
  marginal and Mondrian-by-class on test and shift.
- **Analysis:** bootstrap 95 % CI (2,000 resamples over scenes) of the difference in wrong-accept rate (c) − (a).
  Coverage under shift is reported as measured; no guarantee is claimed there.
