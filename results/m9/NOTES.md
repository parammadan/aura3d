# EXP-M9-001 results (complete): full reliability loop B6-geo vs B1 / B3cal, UR10e, seeds 2000–2049

| Cell | B1 | B3cal | B6-geo | B6 wrong picks | B6 deferred | B6 vs B3cal McNemar |
|---|---|---|---|---|---|---|
| 0°, 0 mm | 50/50 | 50/50 | 50/50 | 0 | 0 | 1.0 |
| 0.5°, 0 mm | 7/50 | 42/50 | 42/50 | 0 | 8 | 1.0 |
| 1°, 0 mm | 2/50 | 30/50 | 30/50 | 0 | 20 | 1.0 |
| 0°, 10 mm | 6/50 | 30/50 | 30/50 | 0 | 20 | 1.0 |

**B6 = B3cal in every cell.** After validated recalibration, no wrong picks remain for the risk gate to stop. All
remaining failures are refused recalibrations (deferrals), which a downstream gate cannot fix. The gate is not
harmful (0 extra deferrals at zero drift), but on this task it adds no closed-loop value. Its measured value is
offline ranking (EXP-M5-003). To show closed-loop value it needs a scenario where geometry is right but individual
estimates are wrong (clutter, occlusion, reflective parts), not yet run with the gate.

# EXP-M9-003 results: risk gate in clutter (seeds 4000–4049, paired; gate off vs on)
| | gate off | gate on |
|---|---|---|
| Episode success | 14/50 (0.28) | 5/50 (0.10) |
| Parts correct | 219/300 | 180/300 |
| **Wrong picks (primary)** | 95 | **34 (−64 %)** |
| Deferred episodes | 0 | 32 |
McNemar on success p = 0.0039, against the gate.

The gate does its job (wrong actions −64 %) but is far too conservative here: the risk model was trained on 3-part,
uncluttered scenes, and 6-part clutter is a shift where its P(out) is miscalibrated (consistent with the conformal
coverage drop to 0.71 under shift, EXP-M5-003). **The gate trades throughput for safety, and with out-of-domain
calibration the trade is poor.** Fix to test: calibration data from the deployment distribution. Not by tuning the
threshold on these test seeds.
