"""PYTHONPATH=src python scripts/make_vision_dataset.py --out datasets/vision_v1 --train 3000 --val 300 --test 300 --shift 300"""
import argparse
from pathlib import Path

from aura3d.vision_training.synth import write_split

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
for s, n in [("train", 3000), ("val", 300), ("test", 300), ("shift", 300)]:
    ap.add_argument(f"--{s}", type=int, default=n)
a = ap.parse_args()
for s in ["train", "val", "test", "shift"]:
    write_split(Path(a.out), s, getattr(a, s))
    print("wrote", s, getattr(a, s))
