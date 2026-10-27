#!/bin/bash
#SBATCH -J aura-wm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/wm_latent_%j.log
# Fallback latent world model (ours). DINOv2 weights pre-cached by a CPU job first.
set -euo pipefail
R=/scratch/madan.pa/aura3d
export TORCH_HOME=$R/torchhub PYTHONPATH=$R/code/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$R/envs/vision/bin/python -m aura3d.world_model_training.train_latent --data $R/datasets/${DATA:-traj_v1} --out $R/runs/${OUT:-wm_latent} ${EXTRA:-}
