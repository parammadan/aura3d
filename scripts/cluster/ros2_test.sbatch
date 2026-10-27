#!/bin/bash
#SBATCH -J aura-ros2
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/ros2_%j.log
# ROS 2 Jazzy (Ubuntu 24.04, python 3.12) in Apptainer on a CPU node. The image has no python3-venv, so Python
# deps go to a user base on scratch (PYTHONUSERBASE). Rendering: OSMesa from the conda mesalib on scratch.
set -euo pipefail
R=/scratch/madan.pa/aura3d
export APPTAINER_CACHEDIR=$R/apptainer_cache
[ -f $R/ros_jazzy.sif ] || apptainer pull $R/ros_jazzy.sif docker://ros:jazzy-ros-base
[ -f $R/get-pip.py ] || curl -sS -o $R/get-pip.py https://bootstrap.pypa.io/get-pip.py
apptainer exec --bind $R --bind /scratch/madan.pa/belief3d/mesa24 $R/ros_jazzy.sif bash -c "
set -e; source /opt/ros/jazzy/setup.bash
export PYTHONUSERBASE=$R/envs/ros_user PATH=$R/envs/ros_user/bin:\$PATH
python3 -m pip --version 2>/dev/null || python3 $R/get-pip.py --user --break-system-packages -q
python3 -m pip install --user --break-system-packages -q --cache-dir $R/pipcache mujoco==3.17.0 scipy pyyaml
cd $R/code
LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib:\${LD_LIBRARY_PATH:-} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  PYTHONPATH=$R/code/src:\$PYTHONPATH python3 tests/integration/ros2_graph_test.py
"
