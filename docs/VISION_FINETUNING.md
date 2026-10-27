# Vision fine-tuning (EXP-M4-001)

TorchVision Mask R-CNN ResNet-50 FPN, COCO_V1 weights (sha256 `bf2d0c1e…`). Synthetic data `datasets/vision_v1`
(`src/aura3d/vision_training/synth.py`) uses domain randomisation: part hue (hue 0–0.75), lighting, table shade, camera
pose jitter (3 cm, 3°), 2–6 parts, 35 % touching-part clutter. Splits are separated by seed range: train 3000,
val 300, test 300, plus a **shift** split of 300 with an unseen hue band (0.75–1.0), brighter light and a different
camera position. Labels come from MuJoCo segmentation (privileged, labels only).

Training (SLURM 11412274, A100-SXM4-40GB): 8 epochs, batch 8, SGD lr 0.02 (backbone 0.1×), warmup + cosine, bf16
autocast, grad clip 10, best checkpoint by val segm mAP, seed 0.

| Model | Trainable params | Test segm mAP | Test bbox mAP | Shift segm mAP | Train img/s | Peak mem | Inference p50 |
|---|---|---|---|---|---|---|---|
| PRE (COCO, no FT, class-agnostic) | 0 | 0.001 | 0.001 | 0.006 | – | – | 25 ms |
| **FT-A head-only** | 17.1 M | 0.854 | 0.906 | **0.858** | **41.1** | **3.9 GB** | 23 ms |
| FT-B layer3/4 + FPN | 42.5 M | 0.853 | 0.922 | 0.852 | 31.4 | 7.3 GB | 23 ms |
| FT-C full | 43.9 M | **0.861** | **0.932** | 0.846 | 25.4 | 13.9 GB | 24 ms |

Reading:
- **COCO weights alone find essentially nothing** (0.001), so fine-tuning is necessary, not optional.
- **FT-A ≈ FT-B ≈ FT-C** within about 0.01 mAP, from one seed per arm, so no difference is established. Full
  fine-tuning buys a slightly tighter box (0.932 vs 0.906) at 0.6× throughput and 3.6× memory. FT-A is the default.
- **The task saturates** (segm AP50 = 1.0 everywhere), and the shift split was *not* harder than test. The shift
  design was too mild to probe domain generalisation. A harder shift (textures, specular metal, real images) is
  needed. Real-data validation (BOP ITODD / IPD) has not been done.
- LoRA is not applied because ResNet-50 is convolutional. A ViT backbone would be needed for a meaningful LoRA arm.
- Latency is A100 batch-1 and says nothing about industrial PCs or Jetson-class hardware.

Downstream effect on the robot (does segmentation quality matter for picking?): EXP-M4-002, clutter.
