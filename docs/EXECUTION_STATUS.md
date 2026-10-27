# Execution status

_Last updated: 2026-10-27_

## Done (evidence in `results/` and `EXPERIMENT_REGISTRY.md`)
| # | Component / experiment | Result |
|---|---|---|
| M1 | Audit, environments, licenses | `docs/*AUDIT*`, `ENVIRONMENT_REPORT.md` |
| M2 | Workcell: UR10e + FANUC M-710iC/50 (official URDF) | Oracle 300/300 each (EXP-M2-001) |
| M3 | Classical perception (B1), closed loop | 100/100, ceiling effect (EXP-M3-001) |
| M4 | Mask R-CNN fine-tuning | COCO 0.001 → 0.85 segm mAP; FT-A ≈ B ≈ C (EXP-M4-001) |
| M4 | Learned segmentation in clutter | 30 → 53/100 (EXP-M4-002) |
| M4 | Collision-aware grasp selection | 50 → **88/100**, = oracle (EXP-M4-003) |
| M5 | Drift monitor + validated recalibration | 0.5°: 7 → 42/50; wrong picks → ~0 (EXP-M5-001) |
| M5 | Grasp tolerance; pose-risk | τ 10–15 mm; AUROC 0.97 vs 0.77; conformal 0.89 → 0.71 under shift (EXP-M5-002/003) |
| M6 | JEPA-WM reproduction | reach-wall 5/12 vs paper 41.6 % |
| M7 | JEPA transfer + fine-tune; fallback WM | zero-shot 2.68× copy → WM-B 0.67×; fallback 0.43× (EXP-M7-001) |
| M8 | WM grasp verification | negative: 99 → 88/100 at 0°; outcome not foreseen (probe 0.72 vs 0.99 on the real frame) (EXP-M8-001/2/3) |
| M9 | Full loop B6; FANUC transfer; gate in clutter | B6 = B3cal; FANUC 6 → 42/50; gate −64/−84 % wrong picks but over-defers (EXP-M9-001…004) |
| – | Recovery state machine, ROS 2 graph test (pass), demo video, paper draft, slides, one-pager | `src/aura3d/recovery`, `docs/ROS2_INTEGRATION.md`, `paper/`, `reports/` |

## Running
- EXP-M9-005 gate + B2 in clutter (SLURM 11415323).

## Not done
Real-image validation (BOP ITODD/IPD, license-limited), gradual-drift and long false-alarm runs, ONNX/TensorRT export,
MoveIt 2, custom ROS messages, informative recovery actions (re-view / singulation), multi-seed training runs.

## Next exact step
Read EXP-M9-005 and fill it into the paper. Then analyse the remaining 12 % clutter failures under B2 + CA.
