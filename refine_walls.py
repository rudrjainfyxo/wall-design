#!/usr/bin/env python3
"""
refine_walls_and_tile.py
------------------------
1. DeepLab (ADE-20K) → wall_mask.png
2. YOLOv8-Seg        → wall00_refined.png, wall01_refined.png, …
3. Tiling            → tiled_main_wall.png (20 × 20 tiles on every wall plane
                       whose centroid falls near the image centre)

Each new run overwrites the PNGs above.

USAGE
-----
python refine_walls_and_tile.py  photo1.jpg  photo2.png  --conf 0.35
"""

import argparse, os, sys, cv2, numpy as np, tarfile, math
import tensorflow as tf
from pathlib import Path

# ─────────── configuration ───────────
LABEL_WALL   = 1
DEEPLAB_TAR  = 'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz'
YOLO_WEIGHTS = 'weights.pt'                              # Roboflow wall model
YOLO_SRC     = Path.home() / 'code' / 'ultralytics'      # cloned ultralytics
CONF_DEF     = 0.4                                       # YOLO confidence
TILE_IMG     = 'tile.jpeg'                               # 1 tile source image
# ──────────────────────────────────────

# Import Ultralytics from local repo (works on Py 3.13 without wheel)
sys.path.insert(0, str(YOLO_SRC))
from ultralytics import YOLO


# ---------- DeepLab minimal loader ----------
class DeepLab:
    IN_NAME = 'ImageTensor:0'
    OUT_NAME = 'SemanticPredictions:0'
    INPUT_SZ = 257

    def __init__(self, tar_path=DEEPLAB_TAR):
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path) as tar:
            for m in tar.getmembers():
                if 'frozen_inference_graph' in m.name:
                    graph_bytes = tar.extractfile(m).read()
                    break
        gdef = tf.compat.v1.GraphDef.FromString(graph_bytes)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(gdef, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def infer(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.resize(bgr, (self.INPUT_SZ, self.INPUT_SZ))[..., ::-1]
        logits = self.sess.run(
            self.OUT_NAME, {self.IN_NAME: [rgb]}
        )[0]                                              # (257 × 257)
        return cv2.resize(
            logits.astype(np.uint8), (w, h), cv2.INTER_NEAREST
        )


# ---------- helpers ----------
def wall_mask_from_deeplab(bgr, dl):
    mask = dl.infer(bgr)
    wall = (mask == LABEL_WALL).astype(np.uint8) * 255
    cv2.imwrite('wall_mask.png', wall)
    return wall.astype(bool)


def refined_plane_masks(img_path, yolo_model, conf_thr, dense_bool):
    res = yolo_model.predict(
        img_path, imgsz=640, conf=conf_thr, iou=0.5, verbose=False
    )[0]
    if res.masks is None:
        return []

    cls = res.boxes.cls.cpu().numpy().astype(int)
    wall_id = {v.lower(): k for k, v in yolo_model.model.names.items()}['wall']
    idxs = np.where(cls == wall_id)[0]

    H, W = dense_bool.shape
    out_files = []
    for k, idx in enumerate(idxs):
        inst = res.masks.data[idx].cpu().numpy()             # 0/1 640 × 640
        inst_up = cv2.resize(inst, (W, H), cv2.INTER_NEAREST).astype(bool)
        refined = (inst_up & dense_bool).astype(np.uint8) * 255
        fname = f'wall{k:02d}_refined.png'
        cv2.imwrite(fname, refined)
        out_files.append(fname)
    return out_files


def tile_central_walls(img_path, mask_files):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    if not mask_files:
        return

    # Central ellipse: radius = 15 % of diagonal
    cx, cy = W / 2, H / 2
    diag2 = H ** 2 + W ** 2
    rad2 = (0.15 ** 2) * diag2

    union = np.zeros((H, W), dtype=bool)
    for mf in mask_files:
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if m is None or m.sum() == 0:
            continue
        M = cv2.moments(m)
        if M['m00'] == 0:
            continue
        mx, my = M['m10'] / M['m00'], M['m01'] / M['m00']
        if (mx - cx) ** 2 + (my - cy) ** 2 <= rad2:
            union |= m.astype(bool)

    if not union.any():
        print('    no wall centroids inside central zone → skip tiling')
        return

    tile = cv2.imread(TILE_IMG)
    if tile is None:
        raise FileNotFoundError(TILE_IMG)

    cell_w, cell_h = math.ceil(W / 20), math.ceil(H / 20)
    pattern = np.tile(
        cv2.resize(tile, (cell_w, cell_h), cv2.INTER_AREA),
        (20, 20, 1)
    )[:H, :W]

    out = img.copy()
    out[union] = pattern[union]
    cv2.imwrite('tiled_main_wall.png', out)


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--conf', type=float, default=CONF_DEF, help='YOLO conf thr')
    args = parser.parse_args()

    deeplab = DeepLab()
    yolo = YOLO(YOLO_WEIGHTS)

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print('❌ missing:', img_path)
            continue

        bgr = cv2.imread(img_path)
        dense_wall_bool = wall_mask_from_deeplab(bgr, deeplab)
        refined_files = refined_plane_masks(
            img_path, yolo, args.conf, dense_wall_bool
        )
        tile_central_walls(img_path, refined_files)

        print(f'✓ {img_path}: saved {len(refined_files)} refined mask(s)')


if __name__ == '__main__':
    main()
