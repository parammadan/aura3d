# EXP-M5-001 results: camera drift dose-response (UR10e, nominal noise, seeds 2000–2049)

Monitor (commissioned on dev seeds 0–19): μ0 = 0.982 mm, σ0 = 0.017 mm, threshold = 1.085 mm.

Episode success = all 3 parts in their class bin. Wrong picks = executed picks that did not place a part. Deferred = recalibration refused (validation failed), so the cell stopped and asked for an operator; this counts as episode failure but is not a wrong action.

| drift (deg, mm) | B1 episodes | B3cal episodes | Δ (paired) | McNemar p | B1 wrong picks | B3cal wrong picks | B3cal deferred eps | B3cal alarm eps |
|---|---|---|---|---|---|---|---|---|
| 0, 0 | 50/50 [0.93, 1.00] | 50/50 [0.93, 1.00] | +0.00 | 1 | 0 | 0 | 0 | 0 |
| 0.25, 0 | 29/50 [0.44, 0.71] | 44/50 [0.76, 0.94] | +0.30 | 0.0015 | 116 | 3 | 5 | 50 |
| 0.5, 0 | 7/50 [0.07, 0.26] | 42/50 [0.71, 0.92] | +0.70 | 5.8e-11 | 251 | 0 | 8 | 50 |
| 1, 0 | 2/50 [0.01, 0.13] | 30/50 [0.46, 0.72] | +0.56 | 5.8e-08 | 285 | 0 | 20 | 50 |
| 2, 0 | 0/50 [0.00, 0.07] | 15/50 [0.19, 0.44] | +0.30 | 6.1e-05 | 295 | 0 | 35 | 50 |
| 0, 5 | 16/50 [0.21, 0.46] | 42/50 [0.71, 0.92] | +0.52 | 2.2e-07 | 184 | 0 | 8 | 50 |
| 0, 10 | 6/50 [0.06, 0.24] | 30/50 [0.46, 0.72] | +0.48 | 3e-06 | 249 | 0 | 20 | 50 |
| 1, 10 | 1/50 [0.00, 0.10] | 22/50 [0.31, 0.58] | +0.42 | 9.5e-07 | 281 | 0 | 28 | 50 |
| 3, 20 | 0/50 [0.00, 0.07] | 7/50 [0.07, 0.26] | +0.14 | 0.016 | 262 | 0 | 43 | 50 |

## Interpretation
- **H2 (partial) supported.** In every drifted cell B3cal beats B1 on paired episode success, with exact McNemar
  p ≤ 0.016, and the effect is largest at 0.5° (+0.70). There were no alarms at zero drift (0/50, 0 false alarms) and
  alarms in 50/50 episodes in every drifted cell, so detection is not the bottleneck at these magnitudes.
- **Failure mode changes from wrong actions to deferrals.** B1 makes 116–295 wrong picks per 50 episodes. B3cal
  makes 3 in total (all at 0.25°), and its remaining failures are deferrals where recalibration did not pass the
  held-out sphere check. Deferrals rise with drift size (5 → 43 of 50).
- **The limiter is recalibration acceptance at larger bumps,** not detection. Candidates: iterate the fit from the
  first correction, more or better-placed reference spheres, robot self-observation. None of this has been tested.
- Not tested here: gradual drift, false-alarm rate over long nominal operation (only 50 episodes ≈ 150 frames), FANUC,
  degraded noise.

# EXP-M5-004: gradual drift + long-run false alarms (seeds 5000–5049, run 2 after the camera-reset bug fix)
- **False alarms: 0 / 1,000 drift-free frames.**
- Linear ramp 0 → 1° over 20 observations: detected in **50/50** runs, at a median of **0.10°** (max 0.35°).
- B1's worst grasp-point error before the first alarm: median 1.07 mm, max **5.2 mm**, below the 10–15 mm tolerance.
  Slow drift is caught before it can cause a bad pick.
