# JEPA-WM MetaWorld reproduction on NU Explorer — report (2026-10-27)

Everything below comes from logs listed in section 8. Nothing is estimated unless it is labelled as an estimate.

## 0. Status
| Item | Status |
|---|---|
| (1) Supported inference (load model, encode a MetaWorld frame, predict 1 step with a real action) | **DONE**: CPU job 11410608; GPU jobs 11410930 (H200) and 11410918 (V100) |
| (2) Official MetaWorld goal-conditioned planning eval, reduced to 12 episodes, `mw-reach-wall` | **DONE**: GPU job 11411060, exit 0. **Success 5/12 = 41.7 %** |
| Deviations from a pure `python -m evals.main` run | 1 code-path substitution (dataset-free preprocessor, §3) and 2 pip pins in the env (§2) |

## 1. Repo, checkpoints, environment
- Repo: `/scratch/madan.pa/aura3d/external/jepa-wms` at commit `13cf1d9` (CC BY-NC 4.0). No files inside the repo were modified. `setup_macros.py` was **not** run, because it writes `macros.py` into the repo and only the plotting utilities import it (`app/plan_common/plot/logs_plan_joint_unif_utils.py`). The eval does not need it.
- HF snapshot: `$SNAP=/scratch/madan.pa/aura3d/hf/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9`. The sha256 values come from download job 11410184 (log `/scratch/madan.pa/aura3d/logs/jepa_env_11410184.log`):
  - `jepa_wm_metaworld.pth.tar` (211,687,551 B): `c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8`. It loads as predictor and proprio-encoder from **epoch 50**, with "All keys matched successfully".
  - `dinov2_vits_224_INet.pth.tar` (3,638,440,915 B): `7bd3a340bfbfb8fb7335d855cab4352812484e53e6449d40fef61bd5107d5a6d`. This is the image decoder head (ViT-L, 303.7 M params, epoch 29). The config points to the URL `.../vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar`. hubconf.py `IMAGE_DECODER_URLS` maps that URL to the HF name `dinov2_vits_224_INet`. I symlinked the HF file to `$TORCH_HOME/hub/checkpoints/vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar`, so torch.hub uses it without downloading again. The decoder is used only for visualisation (`decode_each_iteration`, plots). It does not affect actions or success.
  - Visual encoder: frozen DINOv2 ViT-S/14 (22,056,576 params) from `torch.hub facebookresearch/dinov2` and `dinov2_vits14_pretrain.pth`, cached in `TORCH_HOME=/scratch/madan.pa/aura3d/torchhub`. Its sha256 was not recorded.
- Env vars (`/scratch/madan.pa/aura3d/jepa_runs/env.sh`, sourced in every job): `JEPAWM_DSET=$R/dsets`, `JEPAWM_LOGS=$R/logs`, `JEPAWM_CKPT=$R/ckpt`, `JEPAWM_OSSCKPT=$R/ossckpt` (all under `R=/scratch/madan.pa/aura3d/jepa_runs`), `JEPAWM_HOME=/scratch/madan.pa/aura3d/external` (the README defines it as the directory holding the clone; MetaWorld never uses it), `HF_HOME`, `TORCH_HOME`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONPATH=$REPO`.

## 2. Problems hit and fixes (all inside `/scratch/madan.pa/aura3d/envs/jepa`)
1. **`datasets` and `pyarrow` were incompatible.** Importing `evals.utils` failed with `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` (datasets 2.14.4 with pyarrow 25.0.1). Fix: `uv pip install pyarrow==20.0.0`. Log: `slurm/prep_cpu_11410324.log`.
2. **MuJoCo was too new for this MetaWorld commit.** `metaworld/sawyer_xyz_env.py:134 reset_mocap_welds` failed with `ValueError: could not broadcast input array from shape (11,) into shape (12,)`. In mujoco 3.17.0 `mjNEQDATA` is 12, and this metaworld writes 11 values. I checked the wheel headers: 3.3.7, 3.5.0 and 3.8.0 all have `mjNEQDATA 11`. Fix: `uv pip install mujoco==3.3.7`. Side effect: `dm_control 1.0.50` declares `mujoco>=3.17.0`, so that requirement is now unmet. Nothing in the repo imports `dm_control`. Log: `slurm/smoke_cpu_11410529.log`.
3. The metaworld wheel installed from commit `cdafedf` reports its version as `2.0.0` in package metadata, not 3.0.0. It provides the `*-v3-goal-observable` envs the repo uses.
4. Rendering: EGL works on both CPU node c0674 and GPU nodes. libOSMesa is not installed. At interpreter exit, `mujoco/egl` prints harmless `TypeError: 'NoneType' object is not callable` messages from `__del__`.

## 3. Blocker: gated dataset, and the substitution used
`src/scripts/download_data.py --dataset metaworld` failed with: *"The repository 'facebook/jepa-wms' is gated and requires access approval."* The **dataset** repo (HF datasets/facebook/jepa-wms) is gated. The model repo is not. There was no HF token, and I did not request access.

The official `evals/simu_env_planning/eval.py:main` calls `make_datasets()` (the `METAWORLD_HF` parquet) only to get `action_dim`, `proprio_dim` and the normalisation mean/std. With `goal_source: expert`, the eval does not read the dataset anywhere else. I checked `gc_agent.py` and `plan_evaluator.py`: the dataset is read only when `goal_source` is `dset` or `random_action`. So `scripts/run_eval_with_mem.py` runs the official `evals.main` entrypoint through `runpy`, with exactly one monkeypatch: `evals.simu_env_planning.eval.make_datasets` becomes `scripts/jr_stats.py:make_datasets_from_hub_stats`. That function builds the `Preprocessor` the same way `hubconf.py:_load_model_with_config` does, from the hard-coded `app.plan_common.datasets.DATA_STATS["metaworld"]`, using the same transforms as `evals.utils.make_datasets`. The repo itself supports this path: hubconf says the stats are hard-coded "so datasets do not need to be downloaded". I could not check that these constants are bit-identical to stats computed from the parquet files, because the dataset is gated. To run the eval fully unmodified: get HF access, run `python src/scripts/download_data.py --dataset metaworld`, then run `python -m evals.main --fname <cfg> --debug`.

## 4. Inference smoke test (`scripts/smoke_infer.py`)
The script uses the official `init_module` from `evals.simu_env_planning.eval`, the official `make_env` with `parse_cfg`, and the MetaWorld expert policy. The procedure:
1. Reset `mw-reach-wall` with seed 1 and encode the 224×224 frame plus proprio.
2. Take 5 expert actions, clipped to [-1, 1], and step the env.
3. Normalise the actions and concatenate them into one 20-dim world-model action.
4. Predict one step with `unroll` and compare it with the encoding of the true next frame.

| | CPU (job 11410608, c0674) | **H200** (11410930, d4055) | V100-SXM2-32GB (11410918, d1013)* |
|---|---|---|---|
| model load (incl. 3.6 GB decoder) | 28.0 s | 14.3 s | – |
| encode 1 frame (B=1) | 38.1 ms | **3.33 ms** | 5.19 ms |
| predict 1 WM step (B=1, ctxt 1) | 49.1 ms | **9.74 ms** | 15.7 ms |
| unroll H=6 × B=300 (= 1 CEM iteration) | 138.6 s | **1.31 s** | 3.47 s |
| peak `torch.cuda.max_memory_allocated` | – | 17,057 MB | 6,718 MB |

\*Job 11410918 came from an `sbatch` that printed `Unexpected message received`. I thought the submission had failed and resubmitted as 11410930, but both ran (about 1.5 min each). Both outputs are logged.

Sanity check on H200, in L2 over the 16×16×384 latent:
- prediction with the real action vs the true next latent: **160.3**
- copy of the current frame: 165.1
- zero action: 174.1
- 16 random actions: mean 178.6, min 162.0

The real action's prediction ranks **#1 of 17**, so the model is action-sensitive.

In the first CPU attempt I fed the **unclipped** expert actions (z = 4.9, about 6.6σ after normalisation). The prediction was then *worse* than the copy baseline (268 vs 167). Out-of-range actions break prediction, and an adapter must clip before normalising.

## 5. Planning eval (job 11411060, gpu-short, 1× NVIDIA H200 143 GB, node d4055)
Command (from `slurm/eval_gpu.sbatch`):
```
source /scratch/madan.pa/aura3d/jepa_runs/env.sh; cd $REPO
python $R/scripts/run_eval_with_mem.py --fname $R/configs/repro_mw_reach-wall_eps12.yaml --debug
```
Config `jepa_runs/configs/repro_mw_reach-wall_eps12.yaml` is a copy of the hubconf-registered `configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml`, also saved verbatim as `configs/official_mw_reach-wall.yaml`. Only these fields changed:
- `folder` → `$R/logs/mw_jepa_wm_hf_reach-wall`
- `checkpoint_folder` → `$SNAP`
- `model_kwargs.checkpoint` → `jepa_wm_metaworld.pth.tar`
- `tag`
- `meta.eval_episodes` 48 → **12**

Everything else is unchanged: CEM, 300 samples, 10 elites, 15 iterations, H=6, 3 actions stepped per replan, L2 objective with alpha 0.1, ctxt_window 2, 100 max steps, goal from the expert policy, `succ_def: simu`, seed 1, decoding and plots on.

Results (from `combined_results` and `eval.csv`):
| metric | value |
|---|---|
| **episode_success** | **0.4167 (5/12; successes in ep 0, 2, 5, 7, 11)** |
| ep_expert_succ (expert policy reached goal) | 1.0 (12/12) |
| ep_succ_dist | 0.583 (7/12) |
| episode_reward (mean) | 570.88 |
| ep_end_dist | 0.0903 |
| ep_state_dist | 0.629 |
| ep_time (mean per episode) | 301.8 s |
| total_time (eval) / job wall | 3,622.9 s / 3,649 s (sacct elapsed 01:01:02) |
| planning time per replan (84 replans) | median 22.20 s, mean 19.34 s, min 6.08, max 22.52 (15 CEM iterations ≈ 1.48 s each, including decoding) |
| peak torch max_memory_allocated / reserved | 17,248.1 MB / 19,110.0 MB; nvidia-smi peak 20,050 MiB; mean GPU utilisation 89.7 % |

`ep_total_lpips` and `ep_total_emb_l2` report 0.0, and the `*_xyz`, `*_orientation` and `*_closure` metrics report -1. These are placeholders for metrics not computed for this env. They are not results.

**Comparison with the paper** (arXiv 2512.24497, Tables 2 and 11): JEPA-WM, MW-RW (Reach-Wall), CEM with L2 cost: **41.6 (10.0)**. For reference, DINO-WM gets 35.1 (9.4), and L1 cost gives 40.8.

The setting is **only partially comparable**:
- Same: checkpoint family, planner and hyperparameters (Table 10: N=300, H=6, m=3, K=10, J=15, W=2, f=5, M=100), and expert-goal protocol.
- Different: the paper averages 3 training seeds and several final epochs over about 96 episodes ("e=96 for most environments"). This run is one released checkpoint (epoch 50), one eval seed and 12 episodes.
- Different software: mujoco 3.3.7 and pinned metaworld; the paper's versions are unknown. Also the dataset-free preprocessor (§3).

With n=12, the binomial 95 % CI on 5/12 is roughly 15–72 %. So the near-exact match to 41.6 is consistent with the paper but is not a precise confirmation.

## 6. Interface facts (for a 6-DoF arm action adapter)
**Observation**
- One RGB frame, 224×224, from MetaWorld camera `corner2`, with `cam_pos` overridden to [0.75, 0.075, 0.7] and fovy 60 (default). The frame is flipped vertically after render.
- Preprocessing: uint8 → /255 → ImageNet normalisation, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).
- Encoder: DINOv2 ViT-S/14 → 16×16 patch grid × 384 dims (latent shape [B, T, 1, 16, 16, 384]). No reps normalisation.

**Proprio**
- 4-D: [hand x, y, z (m), gripper opening = finger distance / 0.1 clipped to [0, 1]], taken from `obs[:4]`.
- z-scored with proprio_mean [-0.0029, 0.6544, 0.1541, 0.6402] and std [0.1043, 0.1123, 0.0798, 0.2614].
- Encoded by a linear layer (80 params) to 16 dims and concatenated to every patch token (`proprio_encoding: feature`), so the predictor width is 384+16=400.
- The predictor also predicts the next proprio (`proprio_mode: predict_proprio`). The planning cost uses visual L2 plus 0.1 × proprio L2, at the final step only (`sum_all_diffs: false`).

**Action**
- Env action is 4-D in [-1, 1]: [dx, dy, dz, gripper].
- `set_xyz_action`: mocap position delta = clip(a[:3]) × 0.01 m per env step, i.e. Cartesian end-effector translation only, with no rotation. The mocap is clipped to the box [-0.2, 0.5, 0.06]–[0.2, 0.7, 0.6] m in reach-wall's base class defaults.
- a[3] is the gripper actuator command (±). Each env step runs MuJoCo `frame_skip=5` substeps.
- Actions are z-scored with action_mean [0.0057, 0.1574, -0.1396, 0.1998] and std [0.7359, 0.7341, 0.7183, 0.7432].

**Frame-skip and chunking**
- Dataset frameskip f=5 with `action_skip` 1, so one WM step corresponds to 5 env steps. The WM action is the concatenation of 5 normalised 4-D actions, giving **20-D**. It conditions the predictor through AdaLN (`pred_type: AdaLN`, `action_encoder_inpred`), not through a separate action-encoder module.
- The planner samples in normalised 20-D space with no bounds (`max_norms` unset). It denormalises and reshapes to 5×4 per WM step, and the env clips to [-1, 1].

**Horizon**
- Plans H=6 WM steps, which is 30 env steps (≤ 0.30 m travel per axis).
- Executes the first 3 WM steps (15 env steps), then replans (MPC). An episode is 100 env steps, so 7 replans.
- Predictor context is 2 latent frames (`ctxt_window`). It was trained with `num_hist 3`, 2-step rollout, `local_window_time 3`. Each replan starts from a single current frame.

**Size**
- Predictor: 17,630,480 params (6 AdaLN blocks, 16 heads, RoPE).
- The released checkpoint holds only the predictor and the proprio encoder. The encoder is the stock frozen DINOv2.

## 7. GPU etiquette
- Smoke tests: 20-min limit, 1.5 min used each.
- Eval: 1h40 limit on gpu-short, 1h01 used, GPU about 90 % busy.
- No idle allocations. Nothing is left in the queue from this work.
- All setup, downloads and the CPU model load ran on `short` CPU jobs 11410324, 11410529 and 11410608. A Python import test on the login node was killed by the node (rc 137), so no further work was done there.

## 8. Files
- Report: `/scratch/madan.pa/aura3d/jepa_runs/REPRO_REPORT.md`
- Configs: `jepa_runs/configs/{official_mw_reach-wall.yaml, repro_mw_reach-wall_eps12.yaml}`
- Scripts: `jepa_runs/scripts/{smoke_infer.py, jr_stats.py, run_eval_with_mem.py}`; sbatch files in `jepa_runs/slurm/*.sbatch`
- SLURM logs: `jepa_runs/slurm/{prep_cpu_11410324, smoke_cpu_11410529 (mujoco failure), smoke_cpu_11410608, smoke_gpu_11410918, smoke_gpu_11410930, eval_gpu_11411060}.log`
- Smoke JSON: `jepa_runs/logs/smoke_{cpu,gpu}.json` (the GPU file is from H200 job 11410930)
- Eval outputs: `jepa_runs/logs/mw_jepa_wm_hf_reach-wall/simu_env_planning/reach-wall_L2_cem_H6_nas3_ctxt2_r224_alpha0.1_eps12/`, containing `eval.csv`, `args_eval.yaml`, and `mw-reach-wall/ep_*/` with `visualisation/video_agent_goal_{succ,fail}.mp4`, expert videos and `plan_vis/` decodes
- nvidia-smi trace: `jepa_runs/logs/eval_nvsmi_11411060.csv`
