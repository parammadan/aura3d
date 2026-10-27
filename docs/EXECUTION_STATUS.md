# Execution status

_Last updated: 2026-10-27_

## Done (evidence in repo)
| Component | Evidence |
|---|---|
| Audit + environments | `docs/EXISTING_SYSTEM_AUDIT.md`, `ENVIRONMENT_REPORT.md`, `DEPENDENCY_AND_LICENSE_AUDIT.md` |
| Workcell: UR10e (Menagerie) + FANUC M-710iC/50 (official URDF → MuJoCo), shared gripper, tray, bins, fixed RGB-D camera, reference spheres | `src/aura3d/simulation/workcell.py`, `tests/simulation/test_workcell.py` |
| Robot-agnostic IK/servo, oracle pick-place | EXP-M2-001: **300/300 placed on each robot** (held-out seeds 1000–1099), Wilson 95% [0.987, 1.0]; videos `demos/m2_oracle_*.mp4` (git-ignored) |
| Classical depth perception (B1) + closed-loop sorting | EXP-M3-001: B1 = B0 = **100/100 episodes** on both robots under nominal and degraded stereo noise (ceiling effect) |
| Stereo noise model (OAK-D Pro geometry, assumed σ_d) | `src/aura3d/cameras/noise.py` |
| Camera-drift fault, fixture residual, recalibration with leave-one-sphere-out validation, DriftMonitor | `src/aura3d/calibration/`, EXP-M5-001 registered |
| JEPA-WMs env on cluster | SLURM 11410184 (see ENVIRONMENT_REPORT) |

## In progress
- EXP-M5-001 drift dose-response (B1 vs B3cal), running locally.
- JEPA-WMs official MetaWorld reproduction (cluster, background).
- Vision dataset generator (`vision_training/synth.py`) written and smoke-tested on 7 images; trainer
  (`train_maskrcnn.py`) written, **not yet run**.

## Not started
Pose-risk estimator + conformal calibration, trajectory dataset + action adapter, WM fine-tuning, predictive
planner, recovery state machine, ROS 2 adapters, real-data (BOP) validation, ONNX/TensorRT, paper, demo.

## Known issues
- Condition-number observability gate is not discriminative (measured); acceptance relies on held-out spheres instead.
- Without reference spheres, in-plane camera translation is weakly observable (10 mm only corrected to 7.5 mm).
- Local depth render has limited precision (`ARB_clip_control` unavailable).
- Results JSON written before the first commit records `git: uncommitted` (EXP-M2-001).

## Next exact step
Read EXP-M5-001 results → write the analysis → generate `datasets/vision_v1` → run FT-A/B/C on A100.
