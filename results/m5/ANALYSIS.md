# EXP-M5-001 results: camera drift dose-response (UR10e, nominal noise, seeds 2000–2049)

Monitor (commissioned on dev seeds 0–19): μ0 = 0.982 mm, σ0 = 0.017 mm, threshold = 1.085 mm.

Episode success = all 3 parts in their class bin. Wrong picks = executed picks that did not place a part. Deferred = recalibration refused (validation failed), so the cell stopped and asked for an operator; this counts as episode failure but is not a wrong action.

| drift (deg, mm) | B1 episodes | B3cal episodes | Δ (paired) | McNemar p | B1 wrong picks | B3cal wrong picks | B3cal deferred eps | B3cal alarm eps |
|---|---|---|---|---|---|---|---|---|
| 0, 0 | 50/50 [0.93, 1.00] | 50/50 [0.93, 1.00] | +0.00 | 1 | 0 | 0 | 0 | 0 |
| 0.25, 0 | 29/50 [0.44, 0.71] | 44/50 [0.76, 0.94] | +0.30 | 0.0015 | 116 | 3 | 5 | 50 |
| 0.5, 0 | (missing) | | | | | | | |
| 1, 0 | (missing) | | | | | | | |
| 2, 0 | (missing) | | | | | | | |
| 0, 5 | (missing) | | | | | | | |
| 0, 10 | (missing) | | | | | | | |
| 1, 10 | (missing) | | | | | | | |
| 3, 20 | (missing) | | | | | | | |
