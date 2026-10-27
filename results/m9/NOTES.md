# EXP-M9-001 notes (partial: cells (0,0) and (0.5,0) done; (1,0) and (0,10) running on the cluster)

| Cell | B1 | B3cal | B6-geo (full loop) | B6 wrong picks | B6 deferred | B6 vs B3cal McNemar |
|---|---|---|---|---|---|---|
| 0°, 0 mm | 50/50 | 50/50 | 50/50 | 0 | 0 | 1.0 |
| 0.5°, 0 mm | 7/50 | 42/50 | 42/50 | 0 | 8 | 1.0 |

At 0.5° the risk gate adds nothing over B3cal: once calibration is repaired the remaining failures are refused
recalibrations (deferrals), which no downstream gate can fix. The gate does not hurt nominal operation (0 deferrals).
