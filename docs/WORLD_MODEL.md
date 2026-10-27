# World models

## A. Fallback: independently trained latent dynamics (ours, NOT Meta's JEPA-WM)
`src/aura3d/world_models/latent_dynamics.py`. Frozen DINOv2 ViT-S/14 encoder: z = [CLS, mean patch] (768-d,
standardised). Residual MLP predictor z_{t+1} = z_t + f(z_t, p_t, a_t) with proprio p = TCP xyz + jaw width and
action a = [dx, dy, dz, grip] at 5 Hz. Trained with an 8-step open-loop rollout loss plus a BCE outcome head.

Data `datasets/traj_v1` (UR10e, scripted pick with noise; 35 % of episodes use an 8–20 mm offset grasp):
train 400 episodes (success 0.91), val 60 (0.867), test 60 (0.933); 24 actions per episode.

**EXP-M7-000** (exploratory, not pre-registered; SLURM 11410910, Tesla V100-SXM2-32GB, 3.9 M params, encode 33 s,
train 70 s, 300 epochs). Held-out **test** latent MSE (standardised units) by rollout horizon h:

| h | model | copy z_0 (no dynamics) | model with actions from another episode |
|---|---|---|---|
| 1 | 0.363 | 0.819 | 0.456 |
| 4 | 0.422 | 1.965 | 0.836 |
| 8 | 0.438 | 2.461 | 1.074 |
| 24 | 0.644 | 2.155 | 1.392 |

- The predictor uses the actions: shuffling them raises error 2.5× at h = 8.
- Error grows with horizon (0.36 → 0.64) but stays far below the copy baseline.
- The outcome head's AUROC is 0.93 on val but **0.77 on test, which has only 4 failures**, so it is not a reliable
  estimate. More failure episodes are needed before the head can be used for planning.
- Low latent error does not imply planning value. That is tested separately in M8.

## B. Meta JEPA-WMs (MetaWorld checkpoint)
Reproduction running (cluster). Report pending: `/scratch/madan.pa/aura3d/jepa_runs/REPRO_REPORT.md`.
