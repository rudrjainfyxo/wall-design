#!/usr/bin/env python3
"""
estimate_wall_size_mapanything.py
---------------------------------
Given a UUID produced by wall_api.py, reads

    masks/<UUID>.png          (binary wall mask, 0 / 255)
    masks/<UUID>.(jpg|jpeg|png)   (original RGB photo)

and returns an approximate physical width & height (metres) of that wall
using metric depth from **Map-Anything (ResNet-50 ckpt)**.

Usage:
    python estimate_wall_size_mapanything.py <UUID>
"""

import sys, pathlib, cv2, numpy as np, torch
from PIL import Image

# ───────────────────── locate inputs ─────────────────────
MASK_DIR = pathlib.Path(__file__).parent / "masks"
if len(sys.argv) != 2:
    sys.exit("Usage: python estimate_wall_size_mapanything.py <UUID>")

uid = sys.argv[1].strip()
mask_p = MASK_DIR / f"{uid}.png"
img_p  = next(
    (MASK_DIR / f"{uid}{e}" for e in (".jpg", ".jpeg", ".png")
     if (MASK_DIR / f"{uid}{e}").is_file()),
    None,
)

if not mask_p.is_file() or img_p is None:
    sys.exit("✖  Could not find both mask and original photo for that UUID")

mask_u8 = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
rgb_pil = Image.open(img_p).convert("RGB")

# ───────────────────── load Map-Anything ─────────────────
# helper name changed on 2025-09-25; try both
try:    # newest commits
    from mapanything.utils.hf_utils.helpers import load_inference_model as load_ma
except ImportError:
    try:  # September-2025 beta name
        from mapanything.utils.hf_utils.hf_helpers import load_inference_model as load_ma
    except ImportError:                         # very old commit
        sys.exit("✖  No load_inference_model helper in this mapanything build")

from mapanything.utils.image import load_images            # unchanged since 2024-04

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[Map-Anything] loading ResNet-50 checkpoint … (first run ≈30 s)")
model = load_ma("facebook/map-anything-resnet50", device=device, half=False).eval()

# Build the expected “views” tensor list
views = load_images([rgb_pil])
views[0]["image"] = views[0]["image"].unsqueeze(0).to(device)  # add batch dim

with torch.no_grad():
    pred = model.infer(views)[0]          # first / only image

depth = pred.get("depth_z", pred.get("depth")).squeeze().cpu().numpy()  # H×W
K     = pred["intrinsics"].squeeze().cpu().numpy()                      # 3×3

# ───────────────── back-project mask pixels ─────────────────
ys, xs = np.where(mask_u8 > 0)
if len(xs) < 50:
    sys.exit("✖  Mask too small – aborting")

z  = depth[ys, xs]
xc = (xs - K[0, 2]) * z / K[0, 0]
yc = (ys - K[1, 2]) * z / K[1, 1]
pts = np.stack([xc, yc, z], 1)                        # N × 3  (metres)

# ───────────── fit 2-D bounding box in plane ──────────────
pts_c = pts - pts.mean(0)
_, _, Vt = np.linalg.svd(pts_c, full_matrices=False)   # PCA
proj   = pts_c @ Vt[:2].T                              # N × 2
mins, maxs = proj.min(0), proj.max(0)
width_m, height_m = maxs - mins

print(f"width  ≈ {width_m:5.2f} m")
print(f"height ≈ {height_m:5.2f} m")
