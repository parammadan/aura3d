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

- EXP-M5-001 drift dose-response: B3cal > B1 in every drifted cell (e.g. 0.5°: 42/50 vs 7/50, p = 6e-11); B3cal
  wrong picks 3 total vs B1 116–295 per cell; failures become deferrals (`results/m5/ANALYSIS.md`)
- EXP-M5-002 grasp tolerance τ = block 15, pin 10, bracket 12 mm (dev)
- EXP-M5-003 pose risk: GB AUROC 0.995 (0.966 excluding phantoms) vs 0.754; conformal coverage 0.89 in-dist,
  0.71 under shift (`results/m5/RISK_ANALYSIS.md`)
- Vision dataset `datasets/vision_v1` (3000/300/300/300, 185 MB; uploaded to cluster)

- M6 JEPA-WMs reproduction: reach-wall 5/12 (paper 41.6 %), inference latencies measured (`results/m6/`)
- Fallback latent WM trained: test h=8 MSE 0.438 vs copy 2.461 vs shuffled actions 1.074 (`docs/WORLD_MODEL.md`)

- EXP-M4-001 Mask R-CNN: COCO-only 0.001 segm mAP; FT-A 0.854 / FT-B 0.853 / FT-C 0.861 (no established difference);
  FT-A chosen (1.6× faster, 28 % memory) (`docs/VISION_FINETUNING.md`)
- EXP-M7-001 WM-A: JEPA-WM zero-shot on UR10e = 2.7× worse than copy-z0 at every horizon (transfer fails)
- Recovery state machine with budgets + reason codes (`src/aura3d/recovery/state_machine.py`, tests)
- JEPA adapter (`src/aura3d/world_models/jepa_adapter.py`); traj_v2 (600/100/100, 24 % / 15 % failures in val/test)

## In progress
- EXP-M4-002 clutter B0/B1/B2 (local, CPU Mask R-CNN inference)
- EXP-M7-001 WM-B JEPA predictor fine-tune on traj_v2 (SLURM 11412967); fallback WM v2 (SLURM 11413119)
- EXP-M9-001 full loop B6-geo on drift cells (local)
- EXP-M8-001 WM grasp verification: script + registry ready, waits for WM v2 checkpoint
- Mask R-CNN FT smoke test queued on A100 (SLURM 11410744); trainer not yet run at full scale.
- Written, not run: trajectory recorder (smoke-tested on 3 episodes), fallback latent world model + trainer.

## Not started
ROS 2 adapters, real-data (BOP) validation, ONNX/TensorRT, paper, demo.

## Known issues
- Condition-number observability gate is not discriminative (measured); acceptance relies on held-out spheres instead.
- Without reference spheres, in-plane camera translation is weakly observable (10 mm only corrected to 7.5 mm).
- Local depth render has limited precision (`ARB_clip_control` unavailable).
- Results JSON written before the first commit records `git: uncommitted` (EXP-M2-001).

## Next exact step
Vision smoke test → FT array; record trajectories (train 400 / val 60 / test 60); JEPA repro report → action adapter.
