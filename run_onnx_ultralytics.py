#!/usr/bin/env python3
"""
run_onnx_ultralytics.py – run a Roboflow YOLOv8 ONNX model with Ultralytics,
add class names (or fall back to dummy names), save an annotated image.

Example:
    python run_onnx_ultralytics.py --source 4.jpg --task detect --yaml data.yaml
"""

import argparse, yaml, os, sys
from ultralytics import YOLO

# ───────── helpers ─────────
def load_names(yaml_path: str | None, fallback_classes=100):
    """Return {idx: name}. If YAML missing/unreadable → dummy names."""
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            names = yaml.safe_load(f).get("names", [])
        if isinstance(names, list) and names:
            return {i: n for i, n in enumerate(names)}
    return {i: f"class_{i}" for i in range(fallback_classes)}

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights.onnx", help="ONNX file")
    ap.add_argument("--source", required=True, help="Image to test")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--yaml", default=None, help="Roboflow data.yaml (optional)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"❌ ONNX not found: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"❌ Image not found: {args.source}")

    # 1) load model via Ultralytics
    model = YOLO(args.model, task=args.task)

    # 2) inject class names so Ultralytics doesn’t crash
    model.names = load_names(args.yaml)

    # 3) run inference + save annotated image
    results = model(args.source, imgsz=args.imgsz, conf=args.conf, save=True)
    print("\nDecoded predictions:\n", results[0])

    out_path = results[0].save_dir / results[0].path.name
    print(f"\n✅ Annotated image saved → {out_path}\n")

if __name__ == "__main__":
    main()
