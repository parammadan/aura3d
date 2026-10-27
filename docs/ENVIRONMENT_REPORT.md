# Environment report (verified 2026-10-27)

## Dev machine (local; CPU tests, simulation, classical perception)
- Apple M1, 8 GB RAM, macOS 26.4 (arm64). Ubuntu 24.04 is not available locally.
- `.venv`: Python 3.12.13 (Homebrew), mujoco 3.17.0, numpy 2.5.4, scipy 1.18.1, pyyaml, pytest, xacro, imageio[ffmpeg].
- Rendering: MuJoCo `Renderer` works offscreen via CGL (warning: `ARB_clip_control unavailable`, so depth precision is
  limited; measured residual floor ~1 mm with nominal noise, see docs/CAMERA_CALIBRATION.md).
- No PyTorch locally, deliberately: 8 GB is too small for training, and local LLM/GPU work is off-limits on this machine.

## GPU cluster (NU Explorer; training, JEPA-WMs, vision fine-tuning)
- Access: `ssh explorer`. Partitions seen via `sinfo`: `gpu` (8 h), `gpu-short` (2 h), `gpu-interactive` (2 h),
  `short` (CPU, 2 d), `sharing` (1 h).
- A100 probe (job 11407587, node d1029, A100-SXM4-80GB, driver 570.86.15): NVIDIA EGL vendor file and Vulkan ICD are
  present on GPU nodes and absent on the login node. A100 has no RT cores, so Isaac Sim RTX is not used.
- Modules: cuda 12.1–13.2, cuDNN 9.10.2, miniconda3 24.11.1 / 25.9.1.
- JEPA-WMs env `/scratch/madan.pa/aura3d/envs/jepa`: Python 3.10.22, torch 2.7.0+cu126, torchvision 0.22.0, mujoco 3.17.0,
  metaworld v3.0.0 (`cdafedf`), torchcodec OK (SLURM 11410184). Build script: `scripts/cluster/build_jepa_env.sbatch`.
  - Upstream pins `metaworld@master` (that branch no longer exists) and a malformed d4rl URL. Current Metaworld `main`
    needs Python ≥ 3.11 while jepa-wms needs < 3.11, hence the v3.0.0 pin. d4rl (PointMaze only) is not installed.
  - Conda was run with `--override-channels -c conda-forge`. Anaconda's default-channel terms were not accepted on
    the user's behalf.

## Environment separation
| Env | Where | Purpose |
|---|---|---|
| `.venv` | local | sim, controllers, classical perception, calibration, tests |
| `envs/jepa` | cluster | JEPA-WMs (py3.10, pinned torch) |
| vision env (pending) | cluster | TorchVision Mask R-CNN fine-tuning |
| ROS 2 Jazzy (pending) | needs Ubuntu 24.04 (container or cluster) | integration layer |
