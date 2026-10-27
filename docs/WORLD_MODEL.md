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

## B. Meta JEPA-WMs (MetaWorld checkpoint): official reproduction (M6, done)
Full report: [`results/m6/JEPA_REPRO_REPORT.md`](../results/m6/JEPA_REPRO_REPORT.md).
- Checkpoint `jepa_wm_metaworld.pth.tar` sha256 `c3297772…0bd8` (epoch 50) loads with all keys matched.
- **Inference (H200):** encode 3.33 ms per frame, one predictor step 9.74 ms, one CEM iteration (6 steps × 300 samples)
  1.31 s; peak memory 17.1 GB. The real action's prediction beats copy-z0 and 16 random actions.
- **Official planning eval, reach-wall, CEM + L2, 12 episodes (reduced from 48):** success **5/12 = 41.7 %**
  (binomial 95 % CI ≈ 15–72 %). Paper (arXiv 2512.24497, Tables 2/11): 41.6 ± 10.0 over 3 seeds and ~96 episodes.
  This is **consistent with, not a confirmation of,** the paper. Wall time 61 min (≈ 22 s per replan).
- **Deviations:** the HF MetaWorld dataset is gated and was not downloaded, so normalisation stats came from the
  repo's hubconf constants via a run-time patch (no repo edit). In the env, pyarrow was pinned to 20.0.0 and mujoco to
  3.3.7 for MetaWorld compatibility (the AURA-3D local env still uses 3.17.0). The pinned MetaWorld reports itself as 2.0.0.

### Interface facts that drive the AURA-3D action adapter
| | JEPA-WM MetaWorld | AURA-3D recorder (traj_v1) |
|---|---|---|
| Image | 224² RGB, `corner2` cam, vertically flipped, ImageNet norm | 224² RGB `wm` cam (not flipped) |
| Proprio | 4-D: hand xyz (m) + gripper opening [0, 1], z-scored | TCP xyz + jaw width (m) |
| Action | 4-D in [−1, 1]: dxyz at 1 cm per unit per env step, plus grip | dxyz ≤ 2 cm per 0.2 s step, grip ±1 |
| Chunking | frameskip 5, so 20-D model action | 1 action per model step |
| Planning | horizon 6 model steps, execute 3, replan; context 2 frames | – |

Adapter rules:
1. Rescale our Δxyz to 1 cm units and split each 0.2 s step into 5 sub-steps.
2. Map jaw width to [0, 1].
3. **Clip to the training range before z-scoring**: unclipped actions at ≈ 6.6σ predicted worse than copy-z0.
4. Differences in camera, embodiment and scene mean WM-A (frozen, zero-shot) is expected to transfer poorly. That is
   the hypothesis WM-B (predictor fine-tune) tests.
