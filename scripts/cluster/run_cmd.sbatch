#!/bin/bash
#SBATCH -J aura-cmd
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/cmd_%j.log
# Generic CPU runner: CMD="python scripts/..." sbatch run_cmd.sbatch   (python = envs/sim, OSMesa rendering)
set -euo pipefail
R=/scratch/madan.pa/aura3d
export LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib:${LD_LIBRARY_PATH:-} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export TORCH_HOME=$R/torchhub PYTHONPATH=$R/code/src OMP_NUM_THREADS=2 PATH=$R/envs/sim/bin:$PATH
cd $R/code
echo "$CMD"
eval "$CMD"
