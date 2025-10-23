#!/usr/bin/env python3
"""
refine_walls_1.py  –  YOLO ∩ DeepLab ∪ HQ-SAM wall-mask refinement

ACTIVE OUTPUT
    wall_hqsam_mask_refined.png         # one per input photo
    console:  Rotation  x(pitch)… y(yaw)… z(roll)…   # paste into Unity

TIMING
    • Per-image elapsed seconds
    • Batch total at the end

HOW TO RESTORE TILING
    1. Uncomment the cv2.imwrite() lines in yolo_masks() if you want raw /
       refined mask PNGs again.
    2. Remove “#” on every line of the perspective-tiling block in apply_tiles().
    3. Re-enable the second apply_tiles() call (“yolo” tag) in process_one().
"""

import argparse, os, sys, math, tarfile, cv2, numpy as np, tensorflow as tf
import time
from pathlib import Path

# ───────── CONFIG ─────────────────────────────────────────────────
LABEL_WALL   = 1
DEEPLAB_TAR  = 'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz'
YOLO_WEIGHTS = 'weights.pt'
YOLO_SRC     = Path.home() / 'code' / 'ultralytics'
CONF_DEF     = 0.40
TILE_IMG     = 'tile.jpeg'          # 1-pixel-trimmed tile (tiling only)
FEATHER_PX   = 3
CLEAN_K      = 7
CLEAN_SIGMA  = 1.5
SAM_WEIGHTS  = 'sam_vit_b_01ec64.pth'   # ViT-L backbone (faster on M-series)
MODEL_KEY    = 'vit_b'
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, str(YOLO_SRC))
from ultralytics import YOLO                                # noqa: E402

# ─── HQ-SAM load ────────────────────────────────────────────────
print('[HQ-SAM] loading …')
import torch
from segment_anything import SamPredictor, sam_model_registry

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')        # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')       # NVIDIA
else:
    DEVICE = torch.device('cpu')

sam = sam_model_registry[MODEL_KEY](checkpoint=None)
sam.load_state_dict(torch.load(SAM_WEIGHTS, map_location='cpu'), strict=False)
sam.eval().to(DEVICE)
predictor = SamPredictor(sam)
print('[HQ-SAM] ready on', DEVICE)

# ─── DeepLab loader ──────────────────────────────────────────────
class DeepLab:
    IN_NAME, OUT_NAME, INPUT_SZ = 'ImageTensor:0', 'SemanticPredictions:0', 257
    def __init__(self, tar_path=DEEPLAB_TAR):
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path) as tar:
            buf = next(tar.extractfile(m).read() for m in tar.getmembers()
                       if 'frozen_inference_graph' in m.name)
        gdef = tf.compat.v1.GraphDef.FromString(buf)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(gdef, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)
    def infer(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.resize(bgr, (self.INPUT_SZ, self.INPUT_SZ))[..., ::-1]
        logits = self.sess.run(self.OUT_NAME, {self.IN_NAME: [rgb]})[0]
        return cv2.resize(logits.astype(np.uint8), (w, h), cv2.INTER_NEAREST)

# ─── helpers ──────────────────────────────────────────────────────
def clean_mask(m):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_K, CLEAN_K))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN , k)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=CLEAN_SIGMA)
    return ((m > 127).astype(np.uint8)) * 255

def order_quad(q):
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], np.float32)

# ───────── geometry helpers ──────────────────────────────────────
def wall_quad(mask):
    """Return a robust 4-point quad (TL, TR, BR, BL) around the mask."""
    ys, xs = np.where(mask > 0)
    pts = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    quad = cv2.boxPoints(rect)              # 4×2 float32
    return order_quad(quad.astype(np.float32))


def _homography_to_rotation_and_normal(quad, W, H):
    """Internal: solve homography → R_cam→wall (3×3) and normal."""
    f = max(W, H)
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0,   1  ]], np.float64)

    Hm = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad).astype(np.float64)

    ok, Rs, _, ns = cv2.decomposeHomographyMat(Hm, K)
    if not ok:
        return np.eye(3), np.array([0, 0, -1], np.float32)

    # choose the candidate whose normal faces the camera (nz < 0)
    R_wall2cam, n_cam = next((R, n) for R, n in zip(Rs, ns) if n[2] < 0)

    R_cam2wall = R_wall2cam.T          # invert rotation
    n_cam = R_cam2wall[:, 2]           # Z column is plane normal
    n_cam /= np.linalg.norm(n_cam)
    return R_cam2wall, n_cam.astype(np.float32)


def wall_pose(quad, W, H):
    """
    High-level helper used by FastAPI wrapper.

    Returns:
      pitch, yaw, roll  (degrees, X-Y-Z, Unity order)
      normal            (3-vector, camera space)
    """
    R, n = _homography_to_rotation_and_normal(quad, W, H)

    pitch =  np.degrees(np.arctan2(R[2, 1], R[2, 2]))     # X
    yaw   =  np.degrees(np.arctan2(-R[2, 0],
                                   np.sqrt(R[2, 1]**2 + R[2, 2]**2)))  # Y
    roll  =  np.degrees(np.arctan2(R[1, 0], R[0, 0]))     # Z

    return pitch, yaw, roll, n

# ─── HQ-SAM refine ───────────────────────────────────────────────
def hqsam_refine(photo_bgr, coarse_mask):
    if coarse_mask.sum()==0:
        return coarse_mask
    ys,xs = np.where(coarse_mask>0)
    x0,x1,y0,y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    predictor.set_image(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB))
    masks,_,_ = predictor.predict(box=np.array([[x0,y0,x1,y1]]),
                                  multimask_output=False)
    m0 = masks[0];  m0 = m0.cpu().numpy() if hasattr(m0,"cpu") else m0
    sam_mask = (m0.astype(np.uint8)*255)
    return clean_mask(coarse_mask | sam_mask)

# ─── segmentation helpers ─────────────────────────────────────────
def deeplab_bool(img, dl):
    return (dl.infer(img) == LABEL_WALL)

def yolo_masks(path, yolo, conf, dense_bool):
    res = yolo.predict(path, imgsz=640, conf=conf, iou=0.5, verbose=False)[0]
    if res.masks is None:
        return []
    wall_id = {v.lower():k for k,v in yolo.model.names.items()}['wall']
    idx = np.where(res.boxes.cls.cpu().numpy().astype(int)==wall_id)[0]
    img = cv2.imread(path); H,W = img.shape[:2]
    files=[]
    for k,i in enumerate(idx):
        m = cv2.resize(res.masks.data[i].cpu().numpy(), (W,H),
                       cv2.INTER_NEAREST)*255
        m = clean_mask(m.astype(np.uint8))
        r = clean_mask(((m>0)&dense_bool).astype(np.uint8)*255)
        fname = f'wall{k:02d}_refined.png'
        # cv2.imwrite(fname, r)      # ← enable debug masks
        files.append((fname, r))
    return files

def central_union(named_masks,H,W):
    u = np.zeros((H,W),np.uint8)
    cx,cy,rad2 = W/2, H/2, (0.15**2)*(H*H+W*W)
    for fname,mask in named_masks:
        if mask.sum()==0: continue
        M=cv2.moments(mask); mx,my = M['m10']/M['m00'], M['m01']/M['m00']
        if (mx-cx)**2 + (my-cy)**2 <= rad2: u |= mask
    return u

# ─── tiling routine (currently disabled) ───────────────────────────
def apply_tiles(photo, union_mask, tag):
    if union_mask.sum()==0:
        return
    refined = hqsam_refine(photo, union_mask)
    cv2.imwrite('wall_hqsam_mask_refined.png', refined)

    # -------- Perspective-tiling block (commented) -----------------
    # H, W = photo.shape[:2]
    # quad = wall_quad(refined)
    # grid = tile_canvas(H, W)
    # Hmat = cv2.getPerspectiveTransform(
    #     np.float32([[0, 0],
    #                 [grid.shape[1]-1, 0],
    #                 [grid.shape[1]-1, grid.shape[0]-1],
    #                 [0, grid.shape[0]-1]]),
    #     quad)
    # warp = cv2.warpPerspective(grid, Hmat, (W, H), cv2.INTER_NEAREST,
    #                            borderMode=cv2.BORDER_CONSTANT,
    #                            borderValue=0)
    # warp = lab_L_match(warp, photo, refined)
    # cv2.imwrite(f'tiled_main_wall_persp_{tag}.png',
    #             blend(photo, warp, refined))

    # ---------- Rotation print ------------------------------------
    H, W = photo.shape[:2]
    quad = wall_quad(refined)
    pitch, yaw, roll = plane_angles(quad, W, H)
    print(f'  Rotation  x(pitch)={pitch:+.1f}°  y(yaw)={yaw:+.1f}°  '
          f'z(roll)={roll:+.1f}°')

# ─── pipeline driver with per-image timing ────────────────────────
def process_one(path, dl, yolo, conf):
    t0 = time.perf_counter()
    img = cv2.imread(path)
    if img is None:
        print('❌', path); return
    dense = deeplab_bool(img, dl)
    masks = yolo_masks(path, yolo, conf, dense)
    union = central_union(masks, img.shape[0], img.shape[1])
    apply_tiles(img, union, 'refined')
    # apply_tiles(img, union, 'yolo')   # ← re-enable if needed
    dt = time.perf_counter() - t0
    print(f'✓ {Path(path).name}  ({dt:.2f}s)')

# ─── CLI ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--conf', type=float, default=CONF_DEF)
    args = ap.parse_args()

    dl   = DeepLab()          # load DeepLab once
    yolo = YOLO(YOLO_WEIGHTS) # load YOLO once

    t_all = time.perf_counter()
    for p in args.images:
        process_one(p, dl, yolo, args.conf)
    print(f'\nTotal time for {len(args.images)} image(s): '
          f'{time.perf_counter() - t_all:.2f}s')
