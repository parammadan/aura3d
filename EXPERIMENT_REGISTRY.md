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
