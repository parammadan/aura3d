"""Fine-tune TorchVision Mask R-CNN (ResNet-50 FPN, COCO_V1 weights) on the synthetic workcell data.

Strategies (master prompt §11.2):
  FT-A  head-only: backbone + FPN frozen; RPN and ROI heads trained.
  FT-B  partial:   layer3, layer4 and FPN unfrozen at lr * 0.1; heads at lr.
  FT-C  full:      everything trainable; backbone at lr * 0.1.
  PRE   no training: COCO model evaluated class-agnostically (any detection = "part").
LoRA is not applied: ResNet-50 is convolutional (see docs/VISION_FINETUNING.md).

Engineering: AMP (bf16 on A100), grad-norm clipping, cosine LR with warmup, per-epoch checkpoints with
resume, seed + config hash recorded, throughput and peak GPU memory logged. Run on the cluster:
  python -m aura3d.vision_training.train_maskrcnn --data datasets/vision_v1 --mode FT-A --out runs/ftA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import to_tensor

NUM_CLASSES = 4   # background + block, pin, bracket


class SynthSeg(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, class_agnostic: bool = False):
        self.d = Path(root) / split
        self.index = json.loads((self.d / "index.json").read_text())["images"]
        self.agnostic = class_agnostic

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        e = self.index[k]
        img = to_tensor(Image.open(self.d / "rgb" / e["file"]).convert("RGB"))
        inst = np.array(Image.open(self.d / "inst" / e["file"]))
        masks, boxes, labels = [], [], []
        for j, a in enumerate(e["annotations"], start=1):
            m = inst == j
            if m.sum() == 0:
                continue
            masks.append(m)
            boxes.append(a["bbox_xyxy"])
            labels.append(1 if self.agnostic else a["category_id"])
        h, w = inst.shape
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(labels, dtype=torch.int64),
             "masks": torch.from_numpy(np.stack(masks) if masks else np.zeros((0, h, w), bool)).to(torch.uint8),
             "image_id": torch.tensor(k)}
        return img, t


def build_model(mode: str) -> torch.nn.Module:
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    if mode == "PRE":
        return m
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, NUM_CLASSES)
    for p in m.backbone.parameters():
        p.requires_grad = False
    if mode == "FT-B":
        for name, p in m.backbone.named_parameters():
            if name.startswith(("body.layer3", "body.layer4", "fpn")):
                p.requires_grad = True
    elif mode == "FT-C":
        for name, p in m.backbone.named_parameters():
            if "bn" not in name:           # FrozenBatchNorm stays frozen in torchvision either way
                p.requires_grad = True
    return m


def param_groups(m, lr):
    bb = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("backbone")]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    groups = [{"params": rest, "lr": lr}]
    if bb:
        groups.append({"params": bb, "lr": lr * 0.1})
    return groups


def collate(b):
    return tuple(zip(*b))


@torch.no_grad()
def evaluate(model, ds, device, class_agnostic: bool, max_images: int | None = None) -> dict:
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in ([1] if class_agnostic else [1, 2, 3])]}
    dets, aid, lat = [], 1, []
    n = len(ds) if max_images is None else min(len(ds), max_images)
    for k in range(n):
        img, t = ds[k]
        h, w = img.shape[1:]
        gt["images"].append({"id": k, "height": h, "width": w})
        for b, lab, mk in zip(t["boxes"], t["labels"], t["masks"]):
            rle = mask_util.encode(np.asfortranarray(mk.numpy()))
            rle["counts"] = rle["counts"].decode()
            x0, y0, x1, y1 = b.tolist()
            gt["annotations"].append({"id": aid, "image_id": k, "category_id": 1 if class_agnostic else int(lab),
                                      "bbox": [x0, y0, x1 - x0, y1 - y0], "area": float(mk.sum()), "iscrowd": 0,
                                      "segmentation": rle})
            aid += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model([img.to(device)])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        for b, s, lab, mk in zip(out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu(), out["masks"].cpu()):
            rle = mask_util.encode(np.asfortranarray((mk[0] > 0.5).numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            x0, y0, x1, y1 = b.tolist()
            dets.append({"image_id": k, "category_id": 1 if class_agnostic else int(lab), "score": float(s),
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": rle})
    res = {"n_images": n, "latency_ms_p50": 1000 * float(np.median(lat[3:] or lat)),
           "latency_ms_p95": 1000 * float(np.percentile(lat[3:] or lat, 95))}
    if not dets:
        return {**res, "bbox_mAP": 0.0, "segm_mAP": 0.0, "segm_AP50": 0.0}
    cg = COCO()
    cg.dataset = gt
    cg.createIndex()
    cd = cg.loadRes(dets)
    for iou in ("bbox", "segm"):
        E = COCOeval(cg, cd, iou)
        E.evaluate()
        E.accumulate()
        E.summarize()
        res[f"{iou}_mAP"], res[f"{iou}_AP50"], res[f"{iou}_AP75"] = map(float, E.stats[:3])
        res[f"{iou}_AP_small"] = float(E.stats[3])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["PRE", "FT-A", "FT-B", "FT-C"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(a).items() if k not in ("out", "workers")}
    cfg["torch"], cfg["torchvision"] = torch.__version__, torchvision.__version__
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.mode).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log = {"config": cfg, "config_hash": cfg_hash, "trainable_params": n_train, "total_params": n_total,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "epochs": []}
    agnostic = a.mode == "PRE"
    if a.mode != "PRE":
        ds = SynthSeg(a.data, "train")
        if a.max_train:
            ds = torch.utils.data.Subset(ds, range(min(a.max_train, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                                         drop_last=True, persistent_workers=a.workers > 0)
        opt = torch.optim.SGD(param_groups(model, a.lr), momentum=0.9, weight_decay=1e-4)
        total = a.epochs * len(dl)
        warm = min(300, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        start = 0
        ck = out / "last.pt"
        if ck.exists():                                    # resume after interruption
            st = torch.load(ck, map_location=dev)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            start, log = st["epoch"] + 1, st["log"]
        val = SynthSeg(a.data, "val")
        for ep in range(start, a.epochs):
            model.train()
            t0, seen, losses = time.time(), 0, []
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for imgs, tg in dl:
                imgs = [i.to(dev, non_blocking=True) for i in imgs]
                tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    ld = model(imgs, tg)
                loss = sum(ld.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
                opt.step()
                sched.step()
                seen += len(imgs)
                losses.append({k: float(v) for k, v in ld.items()})
            vres = evaluate(model, val, dev, False, max_images=a.max_eval or 150)
            e = {"epoch": ep, "imgs_per_s": seen / (time.time() - t0),
                 "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
                 "loss": {k: float(np.mean([d[k] for d in losses])) for k in losses[0]},
                 "val_segm_mAP": vres["segm_mAP"], "val_bbox_mAP": vres["bbox_mAP"]}
            log["epochs"].append(e)
            print(json.dumps(e), flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "log": log}, ck)
            best = max(log["epochs"], key=lambda r: r["val_segm_mAP"])
            if best["epoch"] == ep:
                torch.save({"model": model.state_dict(), "config_hash": cfg_hash, "epoch": ep}, out / "best.pt")
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    log["test"] = {s: evaluate(model, SynthSeg(a.data, s, agnostic), dev, agnostic, a.max_eval) for s in ("test", "shift")}
    if a.mode != "PRE":
        log["test_agnostic"] = {s: evaluate(_agnostic(model), SynthSeg(a.data, s, True), dev, True, a.max_eval)
                                for s in ("test", "shift")}
    (out / "results.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"mode": a.mode, "test": log["test"]}, indent=1))


class _agnostic(torch.nn.Module):
    """Wrap a fine-tuned model so every detection is reported as class 1 (for comparison with PRE)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        for o in out:
            o["labels"] = torch.ones_like(o["labels"])
        return out


if __name__ == "__main__":
    main()
