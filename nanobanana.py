#!/usr/bin/env python3
"""
tile_from_pair.py
=================
1. Compare original photo (no tiles) with NanoBanana composite.
2. Generate a clean wall mask via per-pixel comparison.
3. Fill that mask with a 20×20 regular tile grid (no perspective).

Usage
-----
    python tile_from_pair.py 3.jpg nanobanana.png [--tile tile.jpeg]

Requires:  opencv-python, numpy
"""

import cv2, numpy as np, argparse, math, sys, os

# ─────────────────── mask from pixel-wise difference ────────────────────
def derive_mask(orig_bgr, nano_bgr, delta=6):
    """
    Return uint8 mask (255 = wall) using strict per-pixel diff.

    A pixel is marked “changed” if **any** channel differs by > delta.
    delta=6 is tight enough to ignore JPEG noise but catch tile pattern.
    """

    # 1  absolute channel-wise difference
    diff = cv2.absdiff(orig_bgr, nano_bgr)          # uint8 [0..255]

    # 2  flag if any channel jumps > delta
    changed = (diff[:,:,0] > delta) | \
              (diff[:,:,1] > delta) | \
              (diff[:,:,2] > delta)
    mask = (changed.astype(np.uint8) * 255)

    # 3  clean small specks / holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN , k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask

# ─────────────────── build 20×20 grid ───────────────────────────────────
def build_tile_grid(H, W, tile_path):
    tile = cv2.imread(tile_path)
    if tile is None:
        sys.exit(f"❌  cannot read {tile_path}")

    cell_w, cell_h = math.ceil(W/20), math.ceil(H/20)
    cell  = cv2.resize(tile, (cell_w, cell_h), cv2.INTER_AREA)
    grid  = np.tile(cell, (20, 20, 1))[:H, :W]
    return grid

# ─────────────────── soft edge helper ───────────────────────────────────
def feather(mask, px=3):
    return cv2.GaussianBlur(mask.astype(np.float32)/255.0,
                            (0,0), sigmaX=px)[...,None]

# ─────────────────── main program ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('orig', help='original photo (no tiles)')
    ap.add_argument('nano', help='NanoBanana composite')
    ap.add_argument('--tile', default='tile.jpeg',
                    help='single-tile texture (default: tile.jpeg)')
    args = ap.parse_args()

    orig = cv2.imread(args.orig)
    nano = cv2.imread(args.nano)
    if orig is None or nano is None:
        sys.exit("❌  could not read one of the input images")
    if orig.shape != nano.shape:
        nano = cv2.resize(nano, (orig.shape[1], orig.shape[0]))

    H, W = orig.shape[:2]

    # 1  derive crisp mask
    mask = derive_mask(orig, nano, delta=6)
    cv2.imwrite('mask_from_diff.png', mask)
    print("✓  mask_from_diff.png saved")

    # 2  build 20×20 tiled sheet
    grid = build_tile_grid(H, W, args.tile)

    # 3  simple 2-D composite with feathered edge
    alpha = feather(mask, px=3)
    out = orig.astype(np.float32)*(1-alpha) + grid.astype(np.float32)*alpha
    cv2.imwrite('tiled_output.png', out.astype(np.uint8))
    print("✓  tiled_output.png saved")

if __name__ == '__main__':
    main()
