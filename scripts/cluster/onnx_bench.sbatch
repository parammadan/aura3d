#!/bin/bash
#SBATCH -J aura-onnx
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:25:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/onnx_%j.log
set -euo pipefail
R=/scratch/madan.pa/aura3d
export TORCH_HOME=$R/torchhub PYTHONPATH=$R/code/src
cd $R/code
nvidia-smi --query-gpu=name --format=csv,noheader
$R/envs/sim/bin/python scripts/export_onnx_bench.py --ckpt checkpoints/maskrcnn_ftA/best.pt --data $R/datasets/vision_v1
