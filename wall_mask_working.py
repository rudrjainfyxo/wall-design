#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wall-mask prototype (image / video) using TF-1 frozen graph in tar.gz
author: you
"""

import cv2, time, argparse, numpy as np, tarfile, os
import tensorflow as tf
import torch
import math
from torchvision import transforms

LABEL_WALL = 1                                # ADE-20K index for “wall”
MODEL_TAR  = 'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz'

# Load MiDaS once at module import
print("Loading MiDaS small model…")
_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
_midas.eval()
_midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_midas.to(_device)
print("MiDaS ready on", _device)


class DeepLab:
    """Minimal DeepLab loader + inference (MobileNet-V2 ADE-20K)."""
    IN_NAME  = 'ImageTensor:0'
    OUT_NAME = 'SemanticPredictions:0'
    INPUT_SZ = 257                            # 257 × 257 MobileNet-V2

    def __init__(self, tar_path=MODEL_TAR):
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(f'{tar_path} not found')
        # ── extract frozen graph from tar.gz ─────────────────────────────── #
        with tarfile.open(tar_path) as tar:
            graph_bytes = None
            for m in tar.getmembers():
                if 'frozen_inference_graph' in m.name:
                    graph_bytes = tar.extractfile(m).read(); break
            if graph_bytes is None:
                raise RuntimeError('frozen graph not found in tarball')

        graph_def = tf.compat.v1.GraphDef.FromString(graph_bytes)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    # --------------------------------------------------------------------- #
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Return mask (uint8) with ADE-20K class indices, same WxH as input."""
        h, w = bgr.shape[:2]

        # Resize *exactly* to 257×257 (what the graph expects)
        rgb = cv2.resize(bgr, (self.INPUT_SZ, self.INPUT_SZ))
        rgb = rgb[..., ::-1].astype(np.uint8)           # BGR→RGB

        logits = self.sess.run(
            self.OUT_NAME, feed_dict={self.IN_NAME: [rgb]}
        )[0]                                            # (257,257)

        # Back-project to original resolution
        mask = cv2.resize(
            logits.astype(np.uint8), (w, h), cv2.INTER_NEAREST
        )
        return mask
    # --------------------------------------------------------------------- #


def process_image(path: str, model: DeepLab):
    """Run on a single image and show result."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f'Cannot read {path}')

    t0 = time.time()
    mask = model.infer(bgr)
    print(f"Inference time: {time.time() - t0:.2f}s")
   # Build a color mask: white background, green for wall
    h, w = mask.shape
    color_mask = np.ones((h, w, 3), dtype=np.uint8) * 255         # start all white
    color_mask[mask == LABEL_WALL] = (0, 255, 0)                  # wall → green
    cv2.imwrite("wall_mask.png", color_mask)

        # --- compute MiDaS depth map ---
    # Convert BGR→RGB, apply transforms, run model
       # --- compute MiDaS depth map (binary) ---
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_tensor = _midas_transforms(img_rgb).to(_device)            # shape [1,3,H,W]
    with torch.no_grad():
        prediction = _midas(input_tensor)                           # [1,1,H,W]
    depth = prediction.squeeze().cpu().numpy()                      # [H,W] floats

     # Resize to original resolution
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    # ---------- save raw depth (grayscale + colour) ----------
    depth_min, depth_max = depth_resized.min(), depth_resized.max()
    depth_raw_norm = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
    depth_u8 = (depth_raw_norm * 255).astype(np.uint8)

    cv2.imwrite("depth_raw_gray.png", depth_u8)
    cv2.imwrite("depth_raw_color.png",
    cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA))
    # ---------------------------------------------------------
   

    # Normalize to 0–1 float range
    minD, maxD = depth_resized.min(), depth_resized.max()
    depth_norm = (depth_resized - minD) / (maxD - minD + 1e-8)

    # Threshold at brightness 0.1 → binary mask
    bin_mask = (depth_norm > 0.05).astype(np.uint8) * 255           # 0 or 255

    # Optional: convert to 3-channel for consistency
    bin_color = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

    # Save as pure black & white PNG
    cv2.imwrite("depth_map.png", bin_color)


        # ─── combine wall & depth masks ────────────────────────────────────
    # both 'wall_mask.png' and 'depth_map.png' should already be written
    
    # Load the saved masks
    green_white = cv2.imread('wall_mask.png')
    black_white = cv2.imread('depth_map.png')
    if green_white is None or black_white is None:
        raise RuntimeError("Missing 'wall_mask.png' or 'depth_map.png' in cwd.")
    
    # Ensure same size
    h, w = green_white.shape[:2]
    black_white = cv2.resize(black_white, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Prepare output: all white
    common_mask = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Define your colors (BGR)
    GREEN = np.array([0, 255, 0], dtype=np.uint8)
    BLACK = np.array([0, 0, 0], dtype=np.uint8)
    
    # Boolean masks
    is_green = np.all(green_white == GREEN, axis=-1)
    is_black = np.all(black_white == BLACK, axis=-1)
    
    # Where both true → black
    common_mask[np.logical_and(is_green, is_black)] = BLACK
    
    # Save combined result
    cv2.imwrite('common_mask.png', common_mask)
    # ────────────────────────────────────────────────────────────────────

        # ─── smooth edges on the combined mask ─────────────────────────────
    # Load the binary mask (grayscale)
    mask_bin = cv2.imread('common_mask.png', cv2.IMREAD_GRAYSCALE)
    if mask_bin is None:
        raise RuntimeError("Cannot load 'common_mask.png' for smoothing.")

    # Apply Gaussian blur (tune kernel size for more/less smoothing)
    blurred = cv2.GaussianBlur(mask_bin, (35, 5), 0)

    # Re-threshold to get a clean binary mask again
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # (Optional) Morphological opening/closing for extra polish:
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)
    # smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN,  kernel)

    # Convert back to 3-channel BGR if you need color image
    smooth_bgr = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2BGR)

    # Save the smoothed mask
    cv2.imwrite('common_mask_smooth.png', smooth_bgr)
    # ────────────────────────────────────────────────────────────────────

    # ─── overlay smoothed mask on original ─────────────────────────────
    # 'common_mask_smooth.png' should already be written to disk
    mask_sm = cv2.imread("common_mask_smooth.png", cv2.IMREAD_GRAYSCALE)
    if mask_sm is None:
        raise RuntimeError("Missing 'common_mask_smooth.png' for overlay.")

    # Resize mask if it doesn’t match input
    mask_sm = cv2.resize(mask_sm, (w, h), interpolation=cv2.INTER_NEAREST)

    # Build alpha map: white (255) → 0.5 opacity, black (0) → 0
    alpha = (mask_sm.astype(np.float32) / 255.0) * 0.5
    alpha = alpha[:, :, None]  # shape (H, W, 1)

    # Create a white image to overlay
    white_overlay = np.ones_like(bgr, dtype=np.float32) * 255

    # Composite: orig*(1-alpha) + white*alpha
    overlaid = bgr.astype(np.float32) * (1 - alpha) + white_overlay * alpha
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    # Save and/or display
    cv2.imwrite("overlayed_output.png", overlaid)
    cv2.imshow('Overlayed Result', overlaid)
    # ────────────────────────────────────────────────────────────────────

        

    # ─── fill wall region with a 20×20 tile grid ───────────────────────
    tile = cv2.imread('tile.jpeg')
    if tile is None:
        raise RuntimeError("Missing 'tile.jpeg' for tiling.")

    h, w = bgr.shape[:2]

    # Compute each cell’s size using ceil so we never under-tile
    tile_w = math.ceil(w / 20)
    tile_h = math.ceil(h / 20)

    # Resize the single tile into one cell
    tile_cell = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    # Build a grid that’s guaranteed >= image size
    pattern = np.tile(tile_cell, (20, 20, 1))

    # Crop down to exact image dimensions
    pattern = pattern[:h, :w]

    # Make sure your mask_sm is exactly (h, w)
    mask_sm = cv2.resize(mask_sm, (w, h), interpolation=cv2.INTER_NEAREST)

    # Boolean wall region
    wall_pixels = (mask_sm == 0)

    # Apply tiles
    tiled_result = bgr.copy()
    tiled_result[wall_pixels] = pattern[wall_pixels]

    # Save/show
    cv2.imwrite('tiled_output_20x20.png', tiled_result)
    cv2.imshow('Tiled 20×20 Result', tiled_result)
    # ────────────────────────────────────────────────────────────────────


    out = bgr.copy()
    out[mask == LABEL_WALL] = (0, 255, 0)               # highlight walls
    # cv2.imshow('wall mask', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(path: str, out_path: str, model: DeepLab):
    """Stream video, overlay mask, save + preview (ESC to quit)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(
        out_path, fourcc,
        cap.get(cv2.CAP_PROP_FPS) or 30,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    frame_i, t_acc = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- inference every n frames if you want speed boost ----------
        #   n = 1  → every frame
        #   n = 2  → skip half (reuse previous mask)
        n = 1
        if frame_i % n == 0:
            t0 = time.time()
            mask = model.infer(frame)
            t_acc += time.time() - t0

        frame[mask == LABEL_WALL] = (0, 255, 0)
        vw.write(frame)
        cv2.imshow('wall', frame)
        if cv2.waitKey(1) == 27:
            break

        frame_i += 1

    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    if frame_i:
        print(f'Processed {frame_i} frames, mean inf time {t_acc/frame_i:.3f}s')


def main():
    ap = argparse.ArgumentParser(description='wall-mask demo')
    ap.add_argument('src', help='image or video path')
    ap.add_argument('--out', default='out.mp4', help='output video path')
    args = ap.parse_args()

    model = DeepLab()

    if args.src.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(args.src, model)
    else:
        process_video(args.src, args.out, model)


if __name__ == '__main__':
    main()
