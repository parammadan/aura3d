#!/bin/bash
#SBATCH -J aura-vis
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/vis_%x_%A_%a.log
# Array: 0=PRE 1=FT-A 2=FT-B 3=FT-C. Smoke test first:  SMOKE=1 sbatch -t 00:15:00 --array=1 train_vision.sbatch
set -euo pipefail
R=/scratch/madan.pa/aura3d
MODES=(PRE FT-A FT-B FT-C)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
export TORCH_HOME=$R/torchhub PYTHONPATH=$R/code/src
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
EXTRA=""
OUT=$R/runs/vision_v1/$MODE
if [ "${SMOKE:-0}" = "1" ]; then EXTRA="--epochs 1 --max-train 64 --max-eval 20"; OUT=$R/runs/vision_smoke/$MODE; fi
$R/envs/vision/bin/python -m aura3d.vision_training.train_maskrcnn --data $R/datasets/vision_v1 --mode $MODE --out $OUT $EXTRA ${TRAIN_ARGS:-}
