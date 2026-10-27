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
