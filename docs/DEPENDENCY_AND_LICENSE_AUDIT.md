# Dependency and license audit (2026-10-27)

Status: **used** = in the repo or environment now; **planned** = not yet used. "Verify" = license not yet read from
the primary source in this session.

| Component | Version / commit | License | Status | Commercial-use note |
|---|---|---|---|---|
| MuJoCo | 3.17.0 | Apache-2.0 | used | OK |
| MuJoCo Menagerie UR10e | `f3b6013` | BSD-3-Clause (model LICENSE file) | used, vendored in `assets/robots/ur10e` | OK with notice |
| FANUC `fanuc_description` (M-710iC/50) | `fb40c98` (pkg 2.3.0) | Apache-2.0 (package.xml + LICENSES/) | used; xacro expanded, STL visuals | OK with notice; FANUC trademarks are not ours |
| BELIEF-3D | `7976b64` | MIT (same author) | ported files | OK |
| NumPy / SciPy / PyYAML / xacro | 2.5.4 / 1.18.1 / – / PyPI | BSD / BSD / MIT / BSD | used | OK |
| **Meta JEPA-WMs** (code) | `13cf1d9` | **CC BY-NC 4.0** (repo LICENSE) | used (research env) | **Non-commercial only.** It cannot ship in a product without permission from Meta. |
| JEPA-WMs checkpoints (HF `facebook/jepa-wms`) | – | verify on HF model card | used for research | Treat as non-commercial until verified |
| DINOv2 (encoder inside JEPA-WM) | torch hub | Apache-2.0 (verify for the exact weights) | used transitively | likely OK |
| Metaworld | v3.0.0 `cdafedf` | MIT (verify) | used (eval env) | – |
| TorchVision Mask R-CNN | planned | BSD-3 code; COCO-pretrained weights: verify | planned | check weights terms |
| FoundationPose | – | NVIDIA Source Code License (research-only) | not installed | **Non-commercial** |
| FoundationStereo, BundleSDF | – | NVIDIA research-only | not used | Non-commercial |
| BOP ITODD / ITODD-MV | – | verify (believed CC BY-NC-SA 4.0) | planned | Non-commercial if confirmed |
| BOP IPD | – | **conflict**: HF page CC BY 4.0, repo LICENSE CC BY-NC-ND 4.0 | planned | Treat as NC-ND |
| BOP XYZ-IBD | – | CC BY-NC-SA | planned | Non-commercial |
| ROS 2 Jazzy / MoveIt 2 | planned | Apache-2.0 / BSD-3 | planned | OK |
| Open3D | not needed yet | MIT | not installed | OK |

**Consequence for the research claims:** the world-model results (JEPA-WMs) and any FoundationPose / ITODD / IPD /
XYZ-IBD numbers are research-only. A commercially usable version of the reliability layer must rest on the
Apache/BSD/MIT components above. The calibration monitor, risk gate and classical perception already do.
