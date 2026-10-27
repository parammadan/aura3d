#!/bin/bash
#SBATCH -J aura-cpu
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/cpu_%A_%a.log
# CPU-only experiment runner (moved off the 8 GB laptop). OSMesa from BELIEF-3D's conda mesalib.
# Array: 0-3 = EXP-M4-002 clutter shards, 4-5 = EXP-M9-001 cells (1,0) (0,10), 6 = EXP-M8-001 rot 1.0
set -euo pipefail
R=/scratch/madan.pa/aura3d
PY=$R/envs/sim/bin/python
export LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib:${LD_LIBRARY_PATH:-} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export TORCH_HOME=$R/torchhub PYTHONPATH=$R/code/src OMP_NUM_THREADS=4
cd $R/code
i=$SLURM_ARRAY_TASK_ID
case $i in
  0|1|2|3) lo=$((4000 + 25*i)); $PY scripts/run_m4_clutter.py --ckpt $R/code/checkpoints/maskrcnn_ftA/best.pt \
             --seeds $lo:$((lo+25)) --out results/m4/clutter_shard$i.json ;;
  4) $PY scripts/run_m9_full.py --rot 1 --trans 0 ;;
  5) $PY scripts/run_m9_full.py --rot 0 --trans 10 ;;
  6) $PY scripts/run_m8_wm_verify.py --ckpt $R/code/checkpoints/wm_latent_v2/best.pt --rot 1.0 --out results/m8/wm_verify_r1.0.json ;;
esac
