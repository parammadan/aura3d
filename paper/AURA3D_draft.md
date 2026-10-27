# AURA-3D: Validated Calibration Monitoring, Pose-Risk Gating and World-Model Planning for Reliable Simulated Industrial Pick-and-Place

*Draft, not submitted, not peer reviewed. All experiments are in simulation (MuJoCo). Numbers are copied from
`results/`; experiments marked PENDING are still running and nothing here pre-empts them.*

## Abstract
Camera-to-robot calibration drift is a common, silent failure in industrial vision-guided picking. We study it in a
simulated workcell with two embodiments: a UR10e, and a FANUC M-710iC/50 loaded from FANUC's official description. We
build a reliability loop around a classical depth-based localiser. It has three parts: (i) a fixture-residual drift
monitor commissioned on drift-free frames; (ii) robust SE(3) recalibration that is applied only if it predicts a
held-out reference sphere; (iii) a learned pose-risk gate against measured per-class grasp tolerances. In a
pre-registered drift dose-response (50 paired episodes per cell), the monitored system lifts episode success from 7/50
to 42/50 after a 0.5° bump (p = 6×10⁻¹¹) and cuts wrong picks from 251 to 0. Its residual failures are explicit
deferrals. Flat fixtures alone leave camera translation weakly observable, and the information-matrix condition number
does not identify bad fits; held-out validation does. The learned risk model ranks out-of-tolerance estimates far better
than the estimator's own score (AUROC 0.966 vs 0.767 on real-part detections), but its split-conformal bounds lose
coverage under shift (0.71 vs 0.90 nominal). Meta's pretrained JEPA-WM (MetaWorld) does not transfer zero-shot to our
cell (2.7× the error of a copy baseline). An independently trained latent dynamics model predicts well but makes a poor
action verifier: ranking grasp candidates by its success head reduced pick success from 99 % to 88 % at zero drift. We
report this as evidence that world-model value must be measured at the decision, not at the prediction.

## 1 Introduction
- Problem: vision-guided picking fails silently when the believed camera pose is wrong. Confidence scores from the pose
  estimator do not see it.
- Contributions, stated narrowly:
  1. a validated-recalibration protocol (leave-one-sphere-out acceptance) and its dose-response evaluation;
  2. measurements showing *where* the reliability gains come from (wrong actions → deferrals; phantom detections);
  3. a calibrated pose-risk gate with measured, not assumed, task tolerances, and its coverage failure under shift;
  4. a negative result on world-model action verification, with the diagnosis.
- Not claimed: novelty of fixture calibration, conformal prediction or JEPA world models individually; hardware validity.

## 2 Related work (to be completed with verified citations)
Hand-eye calibration and targetless/online recalibration; change detection (CUSUM); pose uncertainty and conformal
prediction for robotics; JEPA / DINO-WM world models for planning (arXiv 2512.24497); BELIEF-3D (our prior simulation
study on calibration-aware active perception).

## 3 Setup
- **Workcell** (`src/aura3d/simulation/`): table, tray, two bins, an OAK-D-Pro-like fixed RGB-D camera (80° HFOV;
  stereo noise σ_z = z²σ_d/(f·b), b = 75 mm, σ_d assumed 0.1 px), three reference spheres on posts, a shared
  parallel-jaw gripper. UR10e (Menagerie) and FANUC M-710iC/50 (FANUC kinematics, limits, inertials; our servo gains).
- **Parts:** block, pin, L-bracket (CAD primitives). **Task:** sort 3 (or 6) parts by class.
- **Seeds:** dev < 1000; every test cell uses registered, disjoint seed ranges (`EXPERIMENT_REGISTRY.md`).

## 4 Methods
- B1 classical localiser: depth → world points → tray/height gates → connected components → height-profile class →
  Huber point-to-SDF planar fit.
- Drift monitor: median |SDF| of static-scene points under the believed extrinsics; alarm at μ0 + 6σ0.
- Recalibration: Huber LSQ on se(3), accepted only if ≥ 2 held-out spheres are each predicted within 2 mm.
- Pose-risk: gradient boosting on observable features → grasp-point error; isotonic map to P(e > τ_cls);
  split/Mondrian conformal bounds. τ_cls from an oracle-offset sweep: block 15, pin 10, bracket 12 mm.
- Recovery state machine with budgets and reason codes (`src/aura3d/recovery/`).
- World models: JEPA-WM MetaWorld (frozen, WM-A; predictor fine-tuned, WM-B) with an explicit action/proprio adapter;
  fallback latent dynamics on frozen DINOv2 with an outcome head; candidate-grasp verifier.

## 5 Results
### 5.1 Controller ceiling (EXP-M2-001)
Oracle poses: UR10e 300/300, FANUC 300/300 placed (Wilson 95 % [0.987, 1]).
### 5.2 Noise alone does not separate methods (EXP-M3-001)
B1 = B0 = 100/100 episodes on both robots, nominal and degraded noise.
### 5.3 Drift dose-response (EXP-M5-001)
See `results/m5/ANALYSIS.md`. B3cal beats B1 in every drifted cell (p ≤ 0.016). B3cal's wrong picks total 3 across all
cells; failures become deferrals as drift grows (5 → 43 of 50). Zero alarms at zero drift (50 episodes).
### 5.4 Observability (docs/CAMERA_CALIBRATION.md)
Without spheres: 10 mm bump → 7.5 mm residual error. With spheres: 1.2 mm. The condition number of JᵀJ was not discriminative.
### 5.5 Pose risk (EXP-M5-003)
Test AUROC 0.995 (GB) vs 0.754 (estimator score); real-part rows only 0.966 vs 0.767 (post hoc). Conformal coverage
0.890 in distribution, 0.71 under shift.
### 5.6 Learned segmentation (EXP-M4-001 / 002)
COCO-only Mask R-CNN: 0.001 segm mAP. Fine-tuned (head-only / partial / full): 0.854 / 0.853 / 0.861, with no
established difference. Downstream, in touching-part clutter (EXP-M4-002, 100 paired seeds), B2 (Mask R-CNN + the same pose fit) sorts
53/100 episodes vs 30/100 for B1 (p = 1.5×10⁻⁵), and first-frame localisation rises from 60.5 % to 99.2 %. The oracle
ceiling is 58/100: in clutter the gripper, not perception, becomes the limit.
### 5.7 Full loop (EXP-M9-001)
B6 = B3cal in every drift cell (0.5°: 42/50; 1°: 30/50; 10 mm: 30/50; 0 wrong picks; failures are deferrals). After
validated recalibration, nothing is left for the risk gate to stop. In clutter (EXP-M9-003) the gate cuts wrong picks by
64 % (95 → 34) but defers 32/50 episodes, lowering success from 14/50 to 5/50 (p = 0.004). It was calibrated on
uncluttered scenes, so this is a calibration-under-shift failure, consistent with §5.5.
### 5.8 Cross-robot transfer (EXP-M9-002)
FANUC M-710iC/50 with no robot-specific code: 0.5° bump B1 6/50 → B3cal 42/50 (p = 3×10⁻¹¹); 10 mm 9/50 → 30/50
(p = 2×10⁻⁴). The camera and fixtures are shared, so this measures modular transfer, not generalisation to a new cell.
### 5.9 World models (EXP-M6/M7/M8)
- JEPA-WM reproduction on its own task: reach-wall 5/12 (paper 41.6 %).
- Zero-shot on our cell: 2.7× copy error. WM-B (predictor fine-tune): **PENDING** (val 0.77× copy after 3 epochs).
- Fallback model: h = 6 error 0.43× copy; outcome head AUROC 0.58 (test).
- Verifier at 0° drift: 88/100 vs geometry 99/100 (p = 0.001); at 1.0°: 16/100 vs 21/100 (p = 0.125); oracle 94/100.
  A retrained, class-balanced outcome head reached val AUROC 0.669, below the pre-set 0.80 gate, so it was not deployed.

## 6 Limitations
Simulation only, with assumed sensor noise. Primitive parts; a mild domain shift for vision; single seeds for training
runs; drift modelled as a rigid bump before the episode. World-model data comes from a scripted policy. JEPA-WMs is
CC BY-NC. Nothing here establishes performance on real FANUC, UR or any vendor's software.

## 7 Conclusion (to be written after the PENDING cells land)
