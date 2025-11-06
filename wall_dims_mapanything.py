#!/usr/bin/env python3
"""
wall_dims_mapanything.py  v1.2
—————————————
Expects inside <folder>:
    <ID>.png        ← original photo  (PNG)
    <ID>_mask.png   ← binary wall-mask (PNG, white = wall)

Example:
    python wall_dims_mapanything.py --id 0282757d99e9484bac787db6edc984ba \
                                    --folder ../masks
"""

import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images


# ─────────────────────────── helpers ────────────────────────────
def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def load_model(dev: str):
    return MapAnything.from_pretrained("facebook/map-anything").to(dev)


@torch.inference_mode()
def get_pts3d(model, img_path: Path, dev: str):
    """Returns H×W×3 numpy array of metric XYZ per-pixel."""
    out = model.infer(load_images([str(img_path)]),
                      use_amp=False,
                      memory_efficient_inference=True)[0]
    return out["pts3d"][0].cpu().numpy()


def longest_span(mask: np.ndarray, axis: int):
    best = (-1, -1, -1, -1)                # (fixed, start, end, length)
    if axis == 1:                          # horizontal → iterate rows
        for r, row in enumerate(mask):
            cols = np.where(row)[0]
            if cols.size:
                L = cols[-1] - cols[0]
                if L > best[3]:
                    best = (r, cols[0], cols[-1], L)
    else:                                  # vertical → iterate cols
        for c in range(mask.shape[1]):
            rows = np.where(mask[:, c])[0]
            if rows.size:
                L = rows[-1] - rows[0]
                if L > best[3]:
                    best = (c, rows[0], rows[-1], L)
    return best[:3]


def dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


# ───────────────────────────── main ─────────────────────────────
def main(p):
    folder = Path(p.folder)
    img_p  = folder / f"{p.id}.png"
    msk_p  = folder / f"{p.id}_mask.png"

    if not img_p.exists() or not msk_p.exists():
        raise FileNotFoundError("Missing image or mask:\n  "
                                f"{img_p}\n  {msk_p}")

    mask_full = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE) > 128
    dev   = pick_device()
    model = load_model(dev)
    pts   = get_pts3d(model, img_p, dev)              # H2 × W2 × 3

    # ---------- resize mask to match pts3d resolution ----------
    H2, W2 = pts.shape[:2]
    mask = cv2.resize(mask_full.astype(np.uint8), (W2, H2),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    assert mask.shape == pts.shape[:2], \
        f"mask {mask.shape}  pts3d {pts.shape[:2]}"
    # -----------------------------------------------------------

    # WIDTH (row with widest span)
    row, c0, c1 = longest_span(mask, axis=1)
    width_m  = dist(pts[row, c0], pts[row, c1])

    # HEIGHT (col with tallest span)
    col, r0, r1 = longest_span(mask, axis=0)
    height_m = dist(pts[r0, col], pts[r1, col])

    out = {"image_id": p.id,
           "width_m":  round(width_m,  3),
           "height_m": round(height_m, 3)}

    print(json.dumps(out, indent=2))
    if p.out: Path(p.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id",     required=True, help="base filename (no ext)")
    ap.add_argument("--folder", default="masks",
                    help="dir holding *.png + *_mask.png")
    ap.add_argument("--out", help="optional JSON output path")
    main(ap.parse_args())
