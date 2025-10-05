#!/usr/bin/env python3
"""
tile_wall_perspective.py – perspective-correct 20×20 tiling with colour-match
"""

import argparse, os, sys, cv2, numpy as np, tarfile, math, tensorflow as tf
from pathlib import Path

# ───────── CONFIG (unchanged) ─────────
LABEL_WALL   = 1
DEEPLAB_TAR  = 'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz'
YOLO_WEIGHTS = 'weights.pt'
YOLO_SRC     = Path.home() / 'code' / 'ultralytics'
CONF_DEF     = 0.4
TILE_IMG     = 'tile.jpeg'
FEATHER_PX   = 3
CLEAN_K      = 7
CLEAN_SIGMA  = 1.5
# ──────────────────────────────────────

sys.path.insert(0, str(YOLO_SRC))
from ultralytics import YOLO


# ───────── basic DeepLab loader (same) ─────────
class DeepLab:
    IN_NAME, OUT_NAME, INPUT_SZ = 'ImageTensor:0','SemanticPredictions:0',257
    def __init__(self, tar_path=DEEPLAB_TAR):
        if not os.path.isfile(tar_path): raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path) as tar:
            gbytes = next(tar.extractfile(m).read()
                          for m in tar.getmembers()
                          if 'frozen_inference_graph' in m.name)
        gdef = tf.compat.v1.GraphDef.FromString(gbytes)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(gdef, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)
    def infer(self, bgr):
        h,w = bgr.shape[:2]
        rgb = cv2.resize(bgr,(self.INPUT_SZ,self.INPUT_SZ))[...,::-1]
        logits = self.sess.run(self.OUT_NAME,{self.IN_NAME:[rgb]})[0]
        return cv2.resize(logits.astype(np.uint8),(w,h),cv2.INTER_NEAREST)


# ───────── util: clean binary mask ─────────
def clean_mask(binary_u8):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLEAN_K,CLEAN_K))
    closed=cv2.morphologyEx(binary_u8,cv2.MORPH_CLOSE,k)
    opened=cv2.morphologyEx(closed,cv2.MORPH_OPEN ,k)
    blur  =cv2.GaussianBlur(opened,(0,0),sigmaX=CLEAN_SIGMA)
    return (blur>127).astype(np.uint8)*255


# ─────────▶️  new: get precise wall quad ─────────
def get_wall_quad(mask):
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt=max(cnts,key=cv2.contourArea)
    peri=cv2.arcLength(cnt,True)
    poly=cv2.approxPolyDP(cnt,0.02*peri,True)
    if len(poly)==4:
        quad=poly.reshape(4,2).astype(np.float32)
    else:
        rect=cv2.minAreaRect(cnt)
        quad=cv2.boxPoints(rect).astype(np.float32)

    s=quad.sum(1); diff=np.diff(quad,axis=1).ravel()
    TL=quad[np.argmin(s)]; BR=quad[np.argmax(s)]
    TR=quad[np.argmin(diff)]; BL=quad[np.argmax(diff)]
    return np.array([TL,TR,BR,BL],np.float32)


# ─────────▶️  new: LAB colour-match ─────────
def lab_match(src_bgr, ref_bgr, mask, strength=1.0):
    """
    Brightness-only harmonisation.
      • strength=1.0  → full L* match
      • strength=0.5  → halfway
      • 0            → disabled
    """
    lab_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_ref = cv2.mean(lab_ref[...,0], mask=mask)[0]
    L_src = cv2.mean(lab_src[...,0], mask=mask)[0]

    lab_src[...,0] += (L_ref - L_src) * strength         # adjust only L*

    return cv2.cvtColor(np.clip(lab_src,0,255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)



# ─────────▶️  new: Poisson or alpha composite ─────────
def composite(base, warped, mask, use_poisson=False):
    if not use_poisson:
        alpha=cv2.GaussianBlur(mask.astype(np.float32)/255.0,
                               (0,0),sigmaX=FEATHER_PX)[...,None]
        return (base*(1-alpha)+warped*alpha).astype(np.uint8)
    center=(mask.shape[1]//2, mask.shape[0]//2)
    return cv2.seamlessClone(warped, base, mask, center,
                             cv2.NORMAL_CLONE)


# ───────── tiling helpers (unchanged) ─────────
def build_tile_grid(H,W):
    tile=cv2.imread(TILE_IMG); tile=tile[1:-1,1:-1]
    cw,ch=math.ceil(W/20),math.ceil(H/20)
    cell=cv2.resize(tile,(cw,ch),cv2.INTER_AREA)
    return np.tile(cell,(20,20,1))[:H,:W]


def wall_mask_from_deeplab(bgr, dl):
    wall=(dl.infer(bgr)==LABEL_WALL).astype(np.uint8)*255
    return wall.astype(bool)


def masks_from_yolo(img_path, yolo, conf_thr, dense_bool):
    res=yolo.predict(img_path,imgsz=640,conf=conf_thr,iou=0.5,verbose=False)[0]
    if res.masks is None: return [],[]
    wall_id={v.lower():k for k,v in yolo.model.names.items()}['wall']
    idxs=np.where(res.boxes.cls.cpu().numpy().astype(int)==wall_id)[0]
    img=cv2.imread(img_path); H,W=img.shape[:2]
    raw_list=[]; ref_list=[]
    for k,i in enumerate(idxs):
        inst=res.masks.data[i].cpu().numpy()
        up=cv2.resize(inst,(W,H),cv2.INTER_NEAREST).astype(np.uint8)*255
        up=clean_mask(up)
        ref=clean_mask(((up>0)&dense_bool).astype(np.uint8)*255)
        rn,fn=f'wall{k:02d}_raw.png',f'wall{k:02d}_refined.png'
        cv2.imwrite(rn,up); cv2.imwrite(fn,ref)
        raw_list.append(rn); ref_list.append(fn)
    return raw_list,ref_list


def central_union(files,H,W):
    cx,cy,rad2=W/2,H/2,(0.15**2)*(H*H+W*W)
    u=np.zeros((H,W),np.uint8)
    for f in files:
        m=cv2.imread(f,0);   # may be None
        if m is None or m.sum()==0: continue
        M=cv2.moments(m); mx,my=M['m10']/M['m00'],M['m01']/M['m00']
        if (mx-cx)**2+(my-cy)**2<=rad2: u|=m
    return u


# ─────────▶️  perspective warp + colour + blend ─────────
def perspective_tile(photo, union_mask, tag, use_poisson):
    if union_mask.sum()==0: return
    H,W=photo.shape[:2]
    quad=get_wall_quad(union_mask)

    grid=build_tile_grid(H,W)
    src=np.array([[0,0],[grid.shape[1]-1,0],
                  [grid.shape[1]-1,grid.shape[0]-1],[0,grid.shape[0]-1]],
                 np.float32)
    Hmat=cv2.getPerspectiveTransform(src, quad)
    warped=cv2.warpPerspective(grid,Hmat,(W,H),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    # colour-match
    warped=lab_match(warped, photo, union_mask)

    out=composite(photo, warped, union_mask, use_poisson)
    cv2.imwrite(f'tiled_main_wall_persp_{tag}.png', out)


def save_outputs(img_path, raw_list, ref_list, use_poisson):
    img=cv2.imread(img_path); H,W=img.shape[:2]
    perspective_tile(img, central_union(ref_list,H,W),'refined',use_poisson)
    perspective_tile(img, central_union(raw_list,H,W),'yolo'   ,use_poisson)


# ───────── main ─────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('images',nargs='+')
    ap.add_argument('--conf',type=float,default=CONF_DEF)
    ap.add_argument('--poisson',action='store_true',
                    help='use seamlessClone instead of alpha blend')
    args=ap.parse_args()

    deeplab=DeepLab(); yolo=YOLO(YOLO_WEIGHTS)

    for p in args.images:
        if not os.path.isfile(p):
            print('❌',p); continue
        dense=wall_mask_from_deeplab(cv2.imread(p),deeplab)
        raw_masks,ref_masks=masks_from_yolo(p,yolo,args.conf,dense)
        save_outputs(p,raw_masks,ref_masks,args.poisson)
        print(f'✓ {p}: perspective tiles saved')

if __name__=='__main__':
    main()
