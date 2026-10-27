#!/bin/bash
#SBATCH -J aura-ros2
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/ros2_%j.log
# ROS 2 Jazzy (Ubuntu 24.04) in Apptainer on a CPU node. MuJoCo renders with OSMesa (software) here: small frames only.
set -euo pipefail
R=/scratch/madan.pa/aura3d
export APPTAINER_CACHEDIR=$R/apptainer_cache
[ -f $R/ros_jazzy.sif ] || apptainer pull $R/ros_jazzy.sif docker://ros:jazzy-ros-base
apptainer exec --bind $R $R/ros_jazzy.sif bash -c "
set -e; source /opt/ros/jazzy/setup.bash
[ -x $R/envs/ros/bin/python ] || python3 -m venv --system-site-packages $R/envs/ros
$R/envs/ros/bin/pip install -q --cache-dir $R/pipcache mujoco==3.17.0 scipy pyyaml PyOpenGL
cd $R/code && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=$R/code/src $R/envs/ros/bin/python tests/integration/ros2_graph_test.py
"
