# EXP-M4-002 results: learned segmentation in touching-part clutter (100 paired seeds 4000–4099)

| Arm | Episode success (6/6) | Wilson 95 % | Parts matched, first frame | Parts correct | Wrong picks |
|---|---|---|---|---|---|
| B0 oracle poses (controller ceiling) | 58/100 | [0.48, 0.67] | 100 % | 89.0 % | 269 |
| B1 classical depth components | 30/100 | [0.22, 0.40] | 60.5 % | 77.2 % | 355 |
| **B2 FT-A Mask R-CNN + same pose fit** | **53/100** | [0.43, 0.62] | **99.2 %** | 86.7 % | 284 |

B2 vs B1: exact McNemar p = 1.5e-5 (discordant 3 vs 26). **Supported:** fine-tuned instance segmentation fixes
merged-blob localisation (60.5 % → 99.2 %) and nearly reaches the controller ceiling (53 vs 58).

The ceiling itself is low: with exact poses only 58 % of 6-part clutter episodes succeed, and 269 picks fail. In
touching clutter the parallel-jaw approach collides with neighbours. That is a grasp-planning limit (collision-aware
grasp selection, singulation), not a perception one.

# EXP-M4-003 results: collision-aware grasp selection (fresh seeds 4200–4299, paired)
| Arm | Episode success | Wilson 95 % | Parts correct | Wrong picks |
|---|---|---|---|---|
| B0 oracle | 62/100 | [0.52, 0.71] | 544/600 | 205 |
| B0 + CA | **88/100** | [0.80, 0.93] | 586/600 | 48 |
| B2 Mask R-CNN | 50/100 | [0.40, 0.60] | 522/600 | 232 |
| B2 + CA | **88/100** | [0.80, 0.93] | 584/600 | 55 |
McNemar CA vs no-CA: B0 p = 8.7e-7 (2 vs 28 discordant); B2 p = 7.5e-11 (1 vs 39).

**Supported.** Choosing the part with the most jaw clearance first (and the best pin yaw), using only estimated poses,
lifts learned perception from 50 to 88/100 and cuts wrong picks by 76 %. **B2 + CA equals the oracle + CA**: with
fine-tuned segmentation plus clearance ordering, perception is no longer the limit in this clutter setting. Remaining
12 % failures are not yet analysed.

# EXP-M4-004: why B2 + CA still fails 12/100 trays (descriptive, seeds rerun with per-pick reasons)
42 failed picks: block motion_blocked 20, bracket motion_blocked 18, block closed_on_nothing 4. **Pins: 0.**
Every remaining failure is a part with a single closing axis whose descent is blocked by a neighbour or wall. The
pin's grasp yaw can rotate, which is why it never fails. Next fix (untested): a second grasp axis for the block (its
60 mm side fits the 110 mm jaw), and a small push / singulation action when no axis is clear.
