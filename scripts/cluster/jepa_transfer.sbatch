#!/bin/bash
#SBATCH -J aura-jepa-tx
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/jepa_tx_%j.log
# EXP-M7-001. Usage: DATA=traj_v2 OUT=m7_jepa [SMOKE=--smoke] sbatch [-t ..] jepa_transfer.sbatch
set -eo pipefail
source /scratch/madan.pa/aura3d/jepa_runs/env.sh
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python /scratch/madan.pa/aura3d/code/scripts/jepa_transfer.py --cfg $R/configs/repro_mw_reach-wall_eps12.yaml \
  --data /scratch/madan.pa/aura3d/datasets/${DATA:-traj_v1} --out /scratch/madan.pa/aura3d/runs/${OUT:-m7_jepa_smoke} ${SMOKE:-}
