# Implementation roadmap (dependency order, not dates)

Each milestone closes only with tests plus a saved result. Status lives in `docs/EXECUTION_STATUS.md`.

| # | Milestone | Depends on | Exit evidence |
|---|---|---|---|
| M1 | Audit + environments | – | 4 audit docs; local venv; cluster JEPA env |
| M2 | Workcell sim, UR10e + FANUC M-710iC/50, oracle pick-place | M1 | EXP-M2-001 JSON, video, sim tests |
| M3 | Camera geometry + classical perception → closed loop (B1) | M2 | EXP-M3-001; projection/segmentation tests |
| M5a | Calibration drift: fault injection, fixture residual monitor, validated recal (B3cal) | M3 | EXP-M5-001 dose-response |
| M6 | JEPA-WMs official reproduction (MetaWorld ckpt) | M1 (cluster) | REPRO_REPORT, measured success rate |
| M4 | Vision fine-tuning: synthetic dataset w/ domain randomisation; Mask R-CNN FT-A/B/C on A100 | M3 | mAP/IoU tables, ckpts, model card; downstream pose effect |
| M5b | Pose-risk estimator: features → P(out-of-tolerance); isotonic / split / Mondrian conformal | M3, M4 | risk-coverage, AUROC, coverage under shift |
| M7 | Trajectory dataset (HDF5) + action adapter; WM-A frozen, WM-B predictor FT; fallback latent-dynamics baseline | M2, M6 | held-out latent loss, action-sensitivity test |
| M8 | Predictive planner (CEM/MPPI over WM) + risk gate; geometric baseline at matched compute | M5b, M7 | B4–B6 vs B1–B3 |
| M9 | Recovery state machine; ROS 2 adapters; cross-robot transfer | M8 | integration tests; FANUC vs UR10e table |
| M10 | Final study: pre-registered baselines + ablations; paper draft; demo | all | registry entries, CIs, manuscript |

Ordering rationale: the reliability layer (M5) is built before the world model because the nominal task is already
at ceiling for B1 (EXP-M3-001: 300/300), so only fault conditions can separate methods.
