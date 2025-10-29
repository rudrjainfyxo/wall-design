#!/usr/bin/env python3
"""
run_onnx_ultralytics.py – runs a Roboflow YOLOv8 ONNX model with Ultralytics,
handles missing class-names gracefully, and saves an annotated image.

ARGS
  --model   Path to ONNX file      (default: weights.onnx)
  --source  Path to input image    (required)
  --task    detect | segment       (default: detect)
  --yaml    Optional YAML with class names (Roboflow data.yaml)

OUTPUT
  • Annotated image in runs/{task}/predict/…
  • Console printout of decoded boxes / masks
"""

import argparse, yaml, os, sys
from ultralytics import YOLO

def load_names(yaml_path, fallback_classes=100):
    """Return dict {idx: name}. If yaml missing → dummy names."""
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            names = yaml.safe_load(f).get("names", [])
        if isinstance(names, list) and names:
            return {i: n for i, n in enumerate(names)}
    # fallback
    return {i: f"class_{i}" for i in range(fallback_classes)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights.onnx")
    ap.add_argument("--source", required=True)
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--yaml", default=None, help="Optional data.yaml with class names")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"❌ ONNX not found: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"❌ Image not found: {args.source}")

    # 1) Load model
    model = YOLO(args.model, task=args.task)
    # 2) Inject class names
    model.model.names = load_names(args.yaml)

    # 3) Run inference
    results = model(args.source, imgsz=args.imgsz, conf=args.conf, save=True)
    print("\nDecoded predictions:\n", results[0])  # prints boxes/masks w/ names

    out_path = results[0].save_dir / results[0].path.name
    print(f"\n✅ Annotated image saved → {out_path}\n")

if __name__ == "__main__":
    main()
