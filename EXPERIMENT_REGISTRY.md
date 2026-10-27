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

## EXP-M4-001: Mask R-CNN fine-tuning strategies (done 2026-10-27)
- PRE (COCO, class-agnostic) vs FT-A/B/C, 8 epochs, seed 0, A100. Test 300 and shift 300 images. Results:
  `results/m4/maskrcnn_*.json`, `docs/VISION_FINETUNING.md`.

## EXP-M4-002: learned segmentation in touching-part clutter (registered 2026-10-27, before test seeds)
- **Question:** does FT-A Mask R-CNN (B2) improve localisation and sorting over classical depth components (B1)
  when parts touch? B0 (oracle poses) is the controller ceiling.
- **Setup:** UR10e, nominal noise, 6 parts (2 per class), min_sep 0.045 m; bin slots; recovery rule "after a failed
  pick, try a different part first" (`skip_failed`) for all arms. Seeds 4000–4099, paired.
- **Primary:** episode success (6/6 sorted). **Secondary:** parts matched on the first frame (right class, grasp-point
  error ≤ 15 mm), parts correct, wrong picks. Exact McNemar B2 vs B1.

## EXP-M8-001: world-model grasp verification under camera drift (registered 2026-10-27, before test seeds)
- **Question (H4, partial):** does ranking perception-derived grasp candidates by a world model's predicted success
  beat executing the perception estimate when the fixed camera has drifted?
- **World model:** fallback LatentDynamics trained on `traj_v2` (`runs/wm_latent_v2`). JEPA-WM is excluded from
  this arm unless WM-B beats the copy baseline on held-out trajectories (EXP-M7-001).
- **Arms:** G (execute estimate), WM (5×5 grid ±16 mm, argmax P(success), defer if < 0.5, ties toward centre),
  OR (oracle). The same reset state and start pose for all arms. Target = largest B1 blob.
- **Conditions:** fixed-camera bump 0° (control) and 1.0° (≈ 17 mm at 1 m, chosen from geometry: beyond the 10–15 mm
  tolerance, inside the grid). Seeds 70000–70099 (never used for trajectories: train 0–599, val 50000+, test 60000+).
- **Primary:** pick success. **Secondary:** wrong actions (executed and failed), deferrals; exact McNemar WM vs G.

## EXP-M9-001: full reliability loop (B6-geo) vs B1 / B3cal (registered 2026-10-27, before running)
- **Arm B6-geo:** AuraRunner = DriftMonitor (same commissioning as EXP-M5-001) + fixture recalibration + GBRisk
  gate (trained on risk_v1 train, isotonic on cal; P(out of tolerance) ≤ 0.5 to execute; ≤ 2 re-observations, then
  defer) + "a part that failed once is not retried". The geometric controller only (no world model; the WM arm
  is EXP-M8-001).
- **Cells:** a subset of the EXP-M5-001 grid: (0,0) (0.5,0) (1,0) (0,10). Seeds 2000–2049, UR10e, nominal noise, so it
  is paired with the stored B1 and B3cal episodes.
- **Primary:** episode success. **Secondary:** wrong picks, deferrals. Exact McNemar B6 vs B3cal.

## EXP-M8-002: verifier with a ranking-validated outcome model (registered 2026-10-27, after EXP-M8-001 control)
- Motivation: EXP-M8-001 control showed a weak outcome head (test AUROC 0.58) degrades picking (88 vs 99/100).
- Change: retrain the fallback model on traj_v2 with class-balanced BCE, outcome weight 1.0 (`runs/wm_latent_v2b`).
- **Authority gate (fixed now):** the verifier is evaluated on EXP-M8-001's protocol and seeds only if val outcome
  AUROC ≥ 0.80. Otherwise it is not deployed, and that is the result.

## EXP-M9-002: transfer of the reliability layer to the FANUC M-710iC/50 (H5) (registered 2026-10-27)
- Same code and configuration as EXP-M5-001 with `--robot fanuc_m710ic_50`. The monitor is re-commissioned on FANUC
  dev seeds 0–19 (same camera). Cells (0,0) (0.5,0) (0,10); seeds 2000–2049; B1 vs B3cal, paired.
- **Primary:** B3cal − B1 episode success per cell (exact McNemar). **Transfer cost:** lines of robot-specific code
  needed beyond `configs/robots.yaml` (counted from git diff).

## EXP-M8-003: patch-level outcome probe on JEPA-WM WM-B (registered 2026-10-27, before running)
- Attention-pooling probe → P(success) on (a) the WM-B *predicted* final latent after unrolling all 24 adapted actions
  from frame 0, and (b) control: the *true* final frame's latent. Class-balanced BCE, 200 epochs, best val AUROC.
- traj_v2 splits. Gate as in EXP-M8-002: (a) val AUROC ≥ 0.80 before any verifier use.
