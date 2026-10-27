"""PERF-001: export FT-A Mask R-CNN to ONNX, check numerical agreement with PyTorch, and benchmark batch-1 latency.

Agreement on 20 test images: matched detections (same label, box IoU >= 0.9), max |score| difference and mask IoU.
Latency: torch fp32 CUDA, ONNX Runtime CUDA EP, and the TensorRT EP if it is available (reported, not assumed).
A100/V100 numbers are datacentre GPU numbers and say nothing about industrial PCs or Jetson.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor

from aura3d.vision_training.train_maskrcnn import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results/perf/onnx_bench.json")
a = ap.parse_args()
dev = torch.device("cuda")
m = build_model("FT-A")
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
m.eval().to(dev)
imgs = sorted(Path(a.data, "test", "rgb").glob("*.png"))[:20]
X = [to_tensor(Image.open(p).convert("RGB")) for p in imgs]
onnx_path = Path(a.out).with_suffix(".onnx")
onnx_path.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
import copy  # noqa: E402

m_cpu = copy.deepcopy(m).cpu().eval()      # torchvision detection export mixes devices on CUDA; export on CPU
torch.onnx.export(m_cpu, ([X[0]],), str(onnx_path), opset_version=17, input_names=["image"],
                  output_names=["boxes", "labels", "scores", "masks"],
                  dynamic_axes={"image": {1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"},
                                "masks": {0: "n"}})
export_s = time.time() - t0
import onnxruntime as ort  # noqa: E402

res = {"export_s": export_s, "onnx_mb": onnx_path.stat().st_size / 1e6, "ort": ort.__version__,
       "providers_available": ort.get_available_providers(), "gpu": torch.cuda.get_device_name(0)}
sess = {"cuda": ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])}
if "TensorrtExecutionProvider" in ort.get_available_providers():
    try:
        sess["tensorrt_fp16"] = ort.InferenceSession(str(onnx_path), providers=[
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        res["tensorrt_error"] = str(e)[:300]

# agreement
agree = []
with torch.no_grad():
    for x in X:
        t = m([x.to(dev)])[0]
        o = sess["cuda"].run(None, {"image": x.numpy()})
        tb, tl, ts = t["boxes"].cpu(), t["labels"].cpu(), t["scores"].cpu()
        ob, ol, os_ = torch.from_numpy(o[0]), torch.from_numpy(o[1]), torch.from_numpy(o[2])
        keep_t, keep_o = ts > 0.5, os_ > 0.5
        n_t, n_o = int(keep_t.sum()), int(keep_o.sum())
        if n_t and n_o:
            iou = box_iou(tb[keep_t], ob[keep_o])
            best, j = iou.max(1)
            same = (tl[keep_t] == ol[keep_o][j]) & (best >= 0.9)
            ds = (ts[keep_t] - os_[keep_o][j]).abs()[same]
            mt = (t["masks"].cpu()[keep_t][:, 0] > 0.5)
            mo = torch.from_numpy(o[3])[keep_o][:, 0] > 0.5
            miou = [float((mt[i] & mo[j[i]]).sum() / max(1, (mt[i] | mo[j[i]]).sum())) for i in range(n_t) if same[i]]
            agree.append({"n_torch": n_t, "n_onnx": n_o, "matched": int(same.sum()),
                          "max_score_diff": float(ds.max()) if len(ds) else None, "mask_iou_min": min(miou) if miou else None})
        else:
            agree.append({"n_torch": n_t, "n_onnx": n_o, "matched": 0})
res["agreement"] = {"images": len(agree), "torch_dets": sum(r["n_torch"] for r in agree),
                    "onnx_dets": sum(r["n_onnx"] for r in agree), "matched": sum(r["matched"] for r in agree),
                    "max_score_diff": max((r.get("max_score_diff") or 0) for r in agree),
                    "min_mask_iou": min((r["mask_iou_min"] for r in agree if r.get("mask_iou_min") is not None), default=None)}


def bench(fn, n=50):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


x0 = X[0]
with torch.no_grad():
    res["latency"] = {"torch_fp32": bench(lambda: m([x0.to(dev)]))}
for k, s in sess.items():
    res["latency"]["ort_" + k] = bench(lambda s=s: s.run(None, {"image": x0.numpy()}))
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
