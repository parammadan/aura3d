#!/bin/bash
#SBATCH -J aura-simenv
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/simenv_%j.log
# envs/sim = venv over the jepa env (torch 2.7) + mujoco 3.17 etc. The jepa env itself is not modified.
set -euo pipefail
R=/scratch/madan.pa/aura3d
[ -x $R/envs/sim/bin/pip ] || $R/envs/jepa/bin/python -m venv --clear --system-site-packages $R/envs/sim
$R/envs/sim/bin/python -m pip install -q --cache-dir $R/pipcache mujoco==3.17.0 scipy pyyaml imageio scikit-learn pycocotools pillow
LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=$R/code/src \
  $R/envs/sim/bin/python -c "
import mujoco, torch, sklearn, scipy; print("mujoco", mujoco.__version__, "torch", torch.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)
from aura3d.simulation.env import WorkcellEnv
e = WorkcellEnv(); e.reset(0); f = e.render('fixed'); print('render ok', f.rgb.shape, float(f.depth.max()))"
$R/envs/jepa/bin/python -c "import mujoco; print('jepa env mujoco unchanged:', mujoco.__version__)"
