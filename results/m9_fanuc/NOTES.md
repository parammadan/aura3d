# EXP-M9-002 results: reliability layer on FANUC M-710iC/50 (seeds 2000–2049, paired)

| Cell | B1 | B3cal | McNemar p |
|---|---|---|---|
| 0°, 0 mm | 50/50 | 50/50 | 1.0 |
| 0.5°, 0 mm | 6/50 | 42/50 | 2.9e-11 |
| 0°, 10 mm | 9/50 | 30/50 | 1.9e-4 |

**Transfer cost: zero robot-specific code.** Only `--robot fanuc_m710ic_50` (configs/robots.yaml) changed. The
re-commissioned monitor threshold is identical (1.085 mm), and B3cal numbers equal the UR10e's cell for cell.
**Caveat:** the camera, fixtures and perception are identical across robots and perception never sees the arm, so this
shows modular transfer (H5), not that the method generalises to a different cell geometry.
