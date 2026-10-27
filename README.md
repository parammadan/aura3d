# AURA-3D: Adaptive, Uncertainty-Aware Robotic Autonomy

Simulation-first research on **reliable industrial pick-and-place under changing workcell conditions**: calibrated
3D perception, camera-drift monitoring with validated recalibration, pose-risk gating, and (in progress)
action-conditioned world models for predictive planning.

> **Status: work in progress.** Everything below is simulation (MuJoCo). Nothing here establishes performance on
> physical hardware. Not affiliated with, and not a reproduction of, any company's product. See
> [`docs/EXECUTION_STATUS.md`](docs/EXECUTION_STATUS.md) for what is done and what is not.

## What works today (measured, held-out seeds)

| Experiment | Result |
|---|---|
| **EXP-M2-001**: oracle pick-place, UR10e and **FANUC M-710iC/50** (official FANUC URDF in MuJoCo) | 300/300 placed on each robot, Wilson 95 % [0.987, 1.0] |
| **EXP-M3-001**: classical depth perception, closed-loop sorting | 100/100 episodes on both robots, nominal and degraded stereo noise (ceiling effect: noise alone does not separate methods) |
| **EXP-M5-001**: camera bump, perception only (B1) vs drift monitor + validated recalibration (B3cal) | 0.5° bump: **7/50 → 42/50** episodes (paired McNemar p = 6e-11). B3cal made 3 wrong picks across all drift cells vs 116–295 per cell for B1; its failures are deferrals, not wrong actions. 0 false alarms at zero drift. |
| **EXP-M5-003**: learned pose-risk vs the estimator's own confidence | AUROC 0.995 vs 0.754 in-distribution (0.966 vs 0.767 excluding phantom detections). Split-conformal coverage 0.89 (target 0.90) in-distribution, **0.71 under shift**, so the guarantee does not survive shift. |

| **EXP-M4-002/003**: touching-part clutter | Fine-tuned Mask R-CNN: 30/100 → 53/100 episodes; + collision-aware grasp selection → **88/100**, equal to oracle poses with the same rule (p = 7.5e-11) |
| **EXP-M9-002**: same code on the FANUC M-710iC/50 | 0.5° bump: 6/50 → 42/50 (p = 3e-11), with zero robot-specific code |
| **EXP-M7/M8**: world models | Meta JEPA-WM: zero-shot 2.7× worse than copy; predictor fine-tune 0.67×. **Neither world model foresees grasp outcome** (probe AUROC 0.72 on the predicted frame vs 0.99 on the real one), so world-model verification is gated off |

Full tables: [`results/m5/ANALYSIS.md`](results/m5/ANALYSIS.md), [`results/m5/RISK_ANALYSIS.md`](results/m5/RISK_ANALYSIS.md).
Experiments are pre-registered in [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) before test seeds are run.

## Findings worth knowing
- **A flat table with low walls leaves camera translation weakly observable.** A 10 mm bump is still 7.5 mm off after
  fixture recalibration; with three reference spheres it is 1.2 mm. The condition number of JᵀJ did *not* separate good
  fits from bad ones, so corrections are accepted only if they predict a held-out sphere within 2 mm.
- **Tilt drift creates phantom parts:** patches of the table pass the height gate. Most of B1's wrong picks under drift
  come from failure modes like this, not from small pose errors (measured grasp tolerance: 10–15 mm).

## Also built
Mask R-CNN fine-tuning (head-only / partial / full); Meta JEPA-WMs reproduction (5/12 vs paper 41.6 %) and adapter;
own latent world model; world-model grasp verifier (negative result); recovery state machine with budgets; ROS 2 Jazzy
nodes + integration test (pass); demo video. Paper draft: `paper/AURA3D_draft.md`. Slides: `reports/`.

## Layout
`src/aura3d/` holds `simulation/`, `robots/`, `controllers/`, `cameras/`, `geometry/`, `pose_estimation/`,
`calibration/`, `uncertainty/`, `vision_training/`, `world_models/` and `evaluation/`. Robot descriptions are in
`configs/robots.yaml`, with assets and their licenses in `assets/robots/`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" scikit-learn imageio[ffmpeg]
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/run_baselines.py --exp demo --baseline B3cal --seeds 2000:2005 --drift-rot 0.5
```

## Licenses
Code: MIT. UR10e model: BSD-3 (MuJoCo Menagerie). FANUC M-710iC/50 description: Apache-2.0 (FANUC-CORPORATION).
**Meta JEPA-WMs is CC BY-NC 4.0**, so world-model results are research-only. Details:
[`docs/DEPENDENCY_AND_LICENSE_AUDIT.md`](docs/DEPENDENCY_AND_LICENSE_AUDIT.md).

Builds on [BELIEF-3D](https://github.com/parammadan/belief-3d) (geometry core and IK, ported with attribution).
