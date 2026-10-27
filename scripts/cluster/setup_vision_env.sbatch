#!/bin/bash
#SBATCH -J aura-vis-env
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/madan.pa/aura3d/logs/vision_env_%j.log
# CPU-only: thin venv over the JEPA env's torch (same torch/torchvision), adds pycocotools, caches COCO weights.
set -euo pipefail
R=/scratch/madan.pa/aura3d
[ -x $R/envs/vision/bin/python ] || $R/envs/jepa/bin/python -m venv --system-site-packages $R/envs/vision
PIP_CACHE_DIR=$R/pipcache $R/envs/vision/bin/pip install -q pycocotools
TORCH_HOME=$R/torchhub $R/envs/vision/bin/python - <<'PY'
import torch, torchvision, pycocotools
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights as W
sd = torch.hub.load_state_dict_from_url(W.COCO_V1.url, progress=False)
print("torch", torch.__version__, "torchvision", torchvision.__version__, "weights tensors", len(sd))
PY
sha256sum $R/torchhub/hub/checkpoints/maskrcnn*
echo DONE
