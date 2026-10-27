#!/bin/bash
#SBATCH -J aura-simenv
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/simenv_%j.log
# Vision venv inherits the jepa env (mujoco 3.3.7 for MetaWorld). AURA-3D sim needs 3.17.0: install it INTO the
# vision venv only (shadows the inherited one); the jepa env is untouched.
set -euo pipefail
R=/scratch/madan.pa/aura3d
$R/envs/vision/bin/python -m pip install -q --cache-dir $R/pipcache mujoco==3.17.0 scipy==1.18.1 pyyaml imageio scikit-learn==1.8.0
$R/envs/vision/bin/python -c "import mujoco, scipy, sklearn; print('mujoco', mujoco.__version__, mujoco.__file__)"
$R/envs/jepa/bin/python -c "import mujoco; print('jepa env mujoco still', mujoco.__version__)"
