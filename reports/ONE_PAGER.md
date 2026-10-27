# AURA-3D: one-page summary

**Goal.** Keep a vision-guided industrial pick cell from acting on wrong geometry, and measure it rigorously.
Simulation (MuJoCo), FANUC M-710iC/50 (official FANUC description) and UR10e. Public: github.com/parammadan/aura3d

**System.** Stereo-depth classical localisation → fixture-residual drift monitor → robust SE(3) recalibration accepted
only on a held-out reference-sphere check → learned pose-risk gate with measured grasp tolerances → bounded recovery
state machine (reason-coded). Learned components: Mask R-CNN (fine-tuned), Meta JEPA-WM (reproduced and adapted), own
latent world model. ROS 2 Jazzy integration test passes.

**Measured (held-out, paired, pre-registered).**
- Drift: 0.5° bump 7/50 → **42/50** episodes, wrong picks 251 → 0; 10 mm bump 6/50 → 30/50; 0 false alarms.
- Observability: 10 mm bump residual 7.5 mm (fixtures only) → 1.2 mm (with reference spheres).
- Pose risk: AUROC 0.97 vs 0.77 (estimator score). Conformal coverage 0.89 in distribution, **0.71 under shift**.
- Vision: COCO-only Mask R-CNN 0.001 segm mAP → 0.85 fine-tuned. Head-only = full within noise, at 28 % of the memory.
- World models: JEPA reproduction 5/12 (paper 41.6 %). Zero-shot on our cell 2.7× worse than copy. Own model as
  verifier **hurt** picking (99 → 88/100), so it is gated off.

**Honest limits.** Simulation only; assumed sensor noise; primitive parts; drift modelled as a rigid bump; single
training seeds. JEPA-WMs is CC BY-NC (research only).

**Pending at time of writing:** clutter (learned vs classical segmentation in closed loop), FANUC transfer, JEPA WM-B
final, remaining full-loop cells.
