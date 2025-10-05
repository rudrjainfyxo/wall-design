#!/usr/bin/env python3
"""
run_wall_masks.py  –  OFF-LINE wall-only instance-seg inference
works inside your existing dlab_env (Python 3.13)

USAGE
-----
python run_wall_masks.py  img1.jpg  img2.png  ...  [--conf 0.35]

OUTPUTS (for each image)
------------------------
<stem>_wall00.png   binary 0/255 mask
<stem>_wall01.png   (etc. if multiple walls)
<stem>_overlay.png  green-tinted preview
"""

import argparse, os, sys, cv2, numpy as np
from pathlib import Path

# --------------------------------------------------------------------------
# 1)  make sure the Ultralytics source tree is importable on Python 3.13
#     (clone once:  git clone https://github.com/ultralytics/ultralytics.git ~/code/ultralytics)
ULTRA_SRC = Path.home() / "code" / "ultralytics"        # change if you cloned elsewhere
sys.path.insert(0, str(ULTRA_SRC))

try:
    from ultralytics import YOLO
except Exception as e:
    print("❌  Ultralytics import failed.  Did you clone the repo to", ULTRA_SRC, "?")
    sys.exit(1)

# --------------------------------------------------------------------------
# 2)  load your local Roboflow-exported weights
MODEL_PATH = "weights.pt"                               # keep alongside this script
if not os.path.isfile(MODEL_PATH):
    print("❌  weights.pt not found next to run_wall_masks.py")
    sys.exit(1)

model = YOLO(MODEL_PATH)                                # CUDA if available, else CPU

# find the class-id for “wall” (case-insensitive)
name_to_id = {name.lower(): idx for idx, name in model.model.names.items()}
WALL_ID = name_to_id.get("wall")
if WALL_ID is None:
    print("❌  'wall' class not present in model names:", model.model.names)
    sys.exit(1)
print(f"✓  Model loaded.  class_id {WALL_ID} == 'wall'")

# --------------------------------------------------------------------------
def process_image(img_path: str, conf_thresh: float):
    stem = Path(img_path).stem
    print(f"[+] {img_path}")

    results = model.predict(img_path, imgsz=640, conf=conf_thresh, iou=0.5, verbose=False)
    r = results[0]                                       # batch size 1

    if r.masks is None:
        print("    no masks predicted"); return

    masks   = r.masks.data.cpu().numpy()                # (N,H,W) 0/1
    classes = r.boxes.cls.cpu().numpy().astype(int)     # length N

    wall_idx = np.where(classes == WALL_ID)[0]
    if not len(wall_idx):
        print("    no wall instances found"); return

    # write each wall mask
    for k, idx in enumerate(wall_idx):
        m = (masks[idx] * 255).astype(np.uint8)
        out_name = f"{stem}_wall{k:02d}.png"
        cv2.imwrite(out_name, m)
        print(f"    saved {out_name}")

    # optional overlay
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    overlay = img.copy()
    for idx in wall_idx:
        m = cv2.resize(masks[idx].astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        overlay[m == 1] = (0, 255, 0)
    vis = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.imwrite(f"{stem}_overlay.png", vis)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="input JPEG/PNG file(s)")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold (0–1)")
    args = ap.parse_args()

    for p in args.images:
        if not os.path.isfile(p):
            print("not found:", p); continue
        process_image(p, conf_thresh=args.conf)
