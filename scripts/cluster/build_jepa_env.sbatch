#!/bin/bash
#SBATCH -J aura-jepa-env
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/jepa_env_%j.log
# CPU-only: builds the isolated JEPA-WMs env and fetches the MetaWorld checkpoint.
set -euo pipefail
ROOT=/scratch/madan.pa/aura3d
module load miniconda3/25.9.1
export CONDA_PKGS_DIRS=$ROOT/condapkgs PIP_CACHE_DIR=$ROOT/pipcache UV_CACHE_DIR=$ROOT/uvcache
export HF_HOME=$ROOT/hf TORCH_HOME=$ROOT/torchhub
ENV=$ROOT/envs/jepa
if [ ! -x $ENV/bin/python ]; then
  conda create -y -p $ENV --override-channels -c conda-forge python=3.10 ffmpeg=7
fi
source activate $ENV || conda activate $ENV
python -m pip install -q uv
cd $ROOT/external/jepa-wms
# Upstream pins metaworld@master (branch no longer exists) and a malformed d4rl URL, so install
# the declared deps ourselves: everything except metaworld/d4rl/mujoco-py helpers, then metaworld
# v3.0.0 (cdafedf; main now needs py>=3.11, jepa-wms needs <3.11), then jepa-wms itself with --no-deps. D4RL is only for PointMaze (unused).
python - <<'PY' > $ROOT/logs/jepa_reqs.txt
import re
block = open("pyproject.toml").read().split("dependencies = [", 1)[1].split("]", 1)[0]
deps = re.findall(r'^\s*"([^"]+)"', block, flags=re.M)   # py3.10 has no tomllib
skip = ("metaworld", "d4rl", "cython", "patchelf")
print("\n".join(d for d in deps if not d.lower().startswith(skip)))
print("pytest\nhuggingface_hub>=0.20.0")
PY
python -m uv pip install --python $ENV/bin/python -r $ROOT/logs/jepa_reqs.txt
python -m uv pip install --python $ENV/bin/python "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@cdafedfaaa896472fb7e76e2fe103e120b67a9b6"
python -m uv pip install --python $ENV/bin/python --no-deps -e .
python -c "import torch, torchvision, mujoco, metaworld; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__, 'mujoco', mujoco.__version__)"
python -c "import torchcodec; print('torchcodec ok')" || echo "torchcodec FAILED"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["jepa_wm_metaworld.pth.tar", "dinov2_vits_224_INet.pth.tar"]:
    p = hf_hub_download(repo_id="facebook/jepa-wms", filename=f)
    print("ckpt", f, p)
PY
python -m pip freeze > $ROOT/logs/jepa_env_freeze.txt
sha256sum $HF_HOME/hub/models--facebook--jepa-wms/snapshots/*/*.pth.tar
echo DONE
