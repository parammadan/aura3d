#!/bin/bash
#SBATCH -J aura-probe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:50:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/jepa_probe_%j.log
# EXP-M8-003. Requires runs/m7_jepa_v2/wmB_predictor.pt (EXP-M7-001 output).
set -eo pipefail
source /scratch/madan.pa/aura3d/jepa_runs/env.sh
cd $REPO
python /scratch/madan.pa/aura3d/code/scripts/jepa_outcome_probe.py --cfg $R/configs/repro_mw_reach-wall_eps12.yaml \
  --data /scratch/madan.pa/aura3d/datasets/traj_v2 --wmb /scratch/madan.pa/aura3d/runs/m7_jepa_v2/wmB_predictor.pt \
  --out /scratch/madan.pa/aura3d/runs/m8_probe
