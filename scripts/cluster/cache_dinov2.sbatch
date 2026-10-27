#!/bin/bash
#SBATCH -J aura-cache
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/cache_dinov2_%j.log
export TORCH_HOME=/scratch/madan.pa/aura3d/torchhub
/scratch/madan.pa/aura3d/envs/vision/bin/python -c "import torch; m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14'); print('ok', sum(p.numel() for p in m.parameters()))"
