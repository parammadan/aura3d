# PERF-001: ONNX export and batch-1 latency of FT-A Mask R-CNN (SLURM 11422829, Tesla V100-SXM2-32GB)
- Export (opset 17, from a CPU copy) 5.1 s, 176 MB. ONNX Runtime 1.23.2 (CUDA and TensorRT EPs available).
- **Agreement on 20 test images: 76/76 detections matched, max score diff 1.8e-6, min mask IoU 1.0.** The export is
  numerically exact.
- **Latency (p50 / p95, batch 1, 640×400): PyTorch fp32 44.4 / 45.1 ms; ORT CUDA 544 / 579 ms; ORT TensorRT fp16
  6,429 / 7,090 ms.**
- ORT is about 12× slower than PyTorch for this torchvision graph, and the TensorRT EP is much worse (the detection graph
  is split into many small TRT subgraphs with fallbacks between them). **Decision: keep PyTorch inference.** A real edge
  path needs a TensorRT-native detector, not an export of torchvision's Mask R-CNN. Datacentre-GPU numbers say nothing
  about industrial PCs or Jetson.
