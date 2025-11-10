#!/usr/bin/env python3
"""
wall_api.py – FastAPI wrapper around two mask generators
Launch:
    uvicorn wall_api:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid, os, shutil, importlib, time, platform
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import psutil
import numpy as np 
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# ─── Optional GPU monitor ────────────────────────────────────────
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
    )
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


# ─── MapAnything singleton ─────────────────────────────────────
_DEV = ("mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")
_MAPANY = MapAnything.from_pretrained("facebook/map-anything").to(_DEV)

@torch.inference_mode()
def _metric_dims(photo_p: Path, mask_p: Path):
    """
    Returns {"width_m": float, "height_m": float} for photo+mask.
    """
    pred = _MAPANY.infer(load_images([str(photo_p)]),
                         use_amp=False,
                         memory_efficient_inference=True)[0]
    pts  = pred["pts3d"][0].cpu().numpy()           # H×W×3

    mask_full = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 128
    H, W = pts.shape[:2]
    mask = cv2.resize(mask_full.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    def span(arr, axis):
        best = (-1, -1, -1, -1)
        if axis:
            for r,row in enumerate(arr):
                cols = np.where(row)[0]
                if cols.size and (L:=cols[-1]-cols[0]) > best[3]:
                    best = (r, cols[0], cols[-1], L)
        else:
            for c in range(arr.shape[1]):
                rows = np.where(arr[:,c])[0]
                if rows.size and (L:=rows[-1]-rows[0]) > best[3]:
                    best = (c, rows[0], rows[-1], L)
        return best[:3]

    def d(a,b): return float(np.linalg.norm(a-b))

    row,c0,c1 = span(mask,1)
    col,r0,r1 = span(mask,0)
    return {
        "width_m":  round(d(pts[row,c0], pts[row,c1]), 3),
        "height_m": round(d(pts[r0,col], pts[r1,col]), 3),
    }

# ─── Camera→wall distance (metric) ────────────────────────────────
@torch.inference_mode()
def _metric_distance(photo_p: Path, mask_p: Path) -> float:
    """
    Finite distance (metres) from camera origin to the wall plane.

    1. Try RANSAC plane fit (needs open3d).
    2. Fallback = robust median-Z.
       If mask has no valid pixels → raise ValueError that bubbles to 404.
    """
    pred = _MAPANY.infer(load_images([str(photo_p)]),
                         use_amp=False,
                         memory_efficient_inference=True)[0]
    pts = pred["pts3d"][0].cpu().numpy()          # H×W×3

    # resize mask → pts3d resolution
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 128
    H, W = pts.shape[:2]
    mask = cv2.resize(mask.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    # ── 1) fast path: RANSAC plane ───────────────────────────────
    try:
        import open3d as o3d
        xyz  = pts[mask]                          # N×3
        if xyz.shape[0] < 50:                     # too few inliers → skip
            raise RuntimeError("mask too small")

        pc   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        _, pl = pc.segment_plane(distance_threshold=0.01,
                                 ransac_n=3, num_iterations=300)
        a, b, c, d = pl
        dist = abs(d) / np.linalg.norm([a, b, c])

    # ── 2) fallback: median-Z ────────────────────────────────────
    except Exception:
        z_vals = pts[..., 2][mask]
        if z_vals.size == 0:
            raise ValueError("Wall mask has no valid depth pixels.")
        dist = float(np.nanmedian(z_vals))
        if not np.isfinite(dist):
            raise ValueError("Depth values are NaN/Inf.")

    return round(float(dist), 3)


# ─── Wrapper class ───────────────────────────────────────────────
class WallRefiner:
    def __init__(self, rw_module):
        self.rw = rw_module
        t0 = time.perf_counter()
        self.dl = rw_module.DeepLab()
        t1 = time.perf_counter()
        self.yolo = rw_module.YOLO(rw_module.YOLO_WEIGHTS)
        t2 = time.perf_counter()

        self.sam_device = (
            rw_module.predictor.model.device.type
            if hasattr(rw_module, "predictor") and hasattr(rw_module.predictor, "model")
            else "unknown"
        )

        self.load_time = {
            "deeplab_load_s": round(t1 - t0, 2),
            "yolo_load_s": round(t2 - t1, 2),
            "total_model_load_s": round(t2 - t0, 2),
        }

        self.debug_info = {
            "torch_cuda_available": torch.cuda.is_available(),
            "sam_device": self.sam_device,
        }

    def run(self, img_path: str):
        times = dict(self.load_time)
        debug = dict(self.debug_info)
        t_total0 = time.perf_counter()

        # ─ image read ─
        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image")
        times["image_read_s"] = round(time.perf_counter() - t0, 2)

        # ─ DeepLab ─
        t1 = time.perf_counter()
        dense = self.rw.deeplab_bool(img, self.dl)
        times["deeplab_infer_s"] = round(time.perf_counter() - t1, 2)

        # ─ YOLO ─
        t2 = time.perf_counter()
        masks = self.rw.yolo_masks(img_path, self.yolo, self.rw.CONF_DEF, dense)
        times["yolo_infer_s"] = round(time.perf_counter() - t2, 2)

        # ─ HQ-SAM refine ─
        t3 = time.perf_counter()
        union = self.rw.central_union(masks, *img.shape[:2])
        refined = self.rw.hqsam_refine(img, union)
        times["sam_refine_s"] = round(time.perf_counter() - t3, 2)

                # ─ Fallback if refined mask area too small ─────────────────────────────
        H, W = img.shape[:2]
        total_pixels = H * W
        refined_area = int((refined > 0).sum())
        ratio = refined_area / total_pixels if total_pixels else 0
        res_fallback = False

        if ratio < 0.01:
            print(f"[WARN] Fallback → refined mask area only {ratio*100:.1f}% of image. Using DeepLab mask.")
            refined = (dense.astype('uint8')) * 255
            res_fallback = True


        if refined.sum() == 0:
            raise ValueError("No wall mask produced")

        # ─ pose calc ─
        t4 = time.perf_counter()
        quad = self.rw.wall_quad(refined)
        pitch, yaw, roll, normal, R = self.rw.wall_pose(quad, img.shape[1], img.shape[0])
        times["pose_compute_s"] = round(time.perf_counter() - t4, 2)

        # ─ save mask ─
        t5 = time.perf_counter()
        out_mask = Path(img_path).with_suffix(".mask.png")
        cv2.imwrite(str(out_mask), refined)
        times["mask_write_s"] = round(time.perf_counter() - t5, 2)

        times["total_pipeline_s"] = round(time.perf_counter() - t_total0, 2)

        # ─ Hardware diagnostics ─
        process = psutil.Process(os.getpid())
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_used_mb = process.memory_info().rss / (1024 * 1024)

        hw_debug = {
            "image_resolution": {"width": img.shape[1], "height": img.shape[0]},
            "cpu_percent": round(cpu_percent, 1),
            "ram_used_mb": round(ram_used_mb, 1),
            "platform": platform.platform()
        }

        if GPU_AVAILABLE:
            util = nvmlDeviceGetUtilizationRates(gpu_handle)
            mem = nvmlDeviceGetMemoryInfo(gpu_handle)
            name = nvmlDeviceGetName(gpu_handle)

            hw_debug["gpu"] = {
                "name": name.decode("utf-8") if isinstance(name, bytes) else str(name),
                "util_percent": util.gpu,
                "mem_used_mb": int(mem.used / 1024 / 1024),
                "mem_total_mb": int(mem.total / 1024 / 1024)
            }

        debug["hardware"] = hw_debug
        debug["fallback_used"] = res_fallback

        return {
            "mask_path": str(out_mask),
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "normal": normal,
            "timings": times,
            "rot_mat":  R.tolist(),
            "debug": debug
        }

# ─── Model modules ───────────────────────────────────────────────
gen_mobile = importlib.import_module("wall_mask_generator_mobile")
# gen_hq     = importlib.import_module("wall_mask_generator")

refiners = {
    "mobile": WallRefiner(gen_mobile),
    # "hq":     WallRefiner(gen_hq),
}

# ─── FastAPI boilerplate ─────────────────────────────────────────
MASK_DIR = Path(__file__).parent / "masks"
MASK_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Wall-Mask Refinement API",
    description="YOLO ∩ DeepLab ∪ HQ-SAM (mobile & HQ) wrapped in FastAPI",
    version="0.6.0",
)

app.mount("/masks", StaticFiles(directory=str(MASK_DIR)), name="masks")


def _handle(file: UploadFile, model_key: str):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="JPEG or PNG only")

    # --- always work with PNG internally ---
    uid = uuid.uuid4().hex
    tmp_upload = f"/tmp/{uid}_upload.png"

    # save uploaded file temporarily
    with open(tmp_upload, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        # run full inference pipeline
        res = refiners[model_key].run(tmp_upload)

        # convert and save original image as PNG (lossless, consistent)
        orig_name = f"{uid}.png"
        img = cv2.imread(tmp_upload)
        if img is None:
            raise ValueError("Cannot read uploaded image for saving.")
        cv2.imwrite(str(MASK_DIR / orig_name), img)

    except Exception as exc:
        if os.path.exists(tmp_upload):
            os.remove(tmp_upload)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # cleanup temp upload
        if os.path.exists(tmp_upload):
            os.remove(tmp_upload)

    # save refined mask next to original image with _mask suffix
    mask_name = f"{uid}_mask.png"
    shutil.move(res["mask_path"], MASK_DIR / mask_name)

    # optional wall-size values (only if present in res)
    wall_size = (
        {"width": res["width_m"], "height": res["height_m"]}
        if "width_m" in res and "height_m" in res
        else None
    )

    # --- build final API response ---
    return {
        "id": uid, 
        "rotation_matrix_cam": res["rot_mat"],
        "rotation": {
            "pitch": round(res["pitch"], 2),
            "yaw":   round(res["yaw"],   2),
            "roll":  round(res["roll"],  2),
        },
        "wall_normal": {
            "x": round(float(res["normal"][0]), 4),
            "y": round(float(res["normal"][1]), 4),
            "z": round(float(res["normal"][2]), 4),
        },
        "fallback_used": res["debug"].get("fallback_used", False),
        **({"wall_size_m": wall_size} if wall_size else {}),
        "original_url": f"/masks/{uid}.png",         # ← always PNG
        "mask_url":     f"/masks/{uid}_mask.png",    # ← always _mask.png
        "timings_s":    res["timings"],
        "debug":        res["debug"],
    }



# ─── API endpoints ───────────────────────────────────────────────
@app.post("/process_mobile")
async def process_mobile(file: UploadFile = File(...)):
    "Runs the lightweight ViT-B SAM model."
    return _handle(file, "mobile")

# @app.post("/process_hq")
# async def process_hq(file: UploadFile = File(...)):
#     "Runs the heavier HQ ViT-L SAM model."
#     return _handle(file, "hq")

@app.get("/measure/{image_id}")
@app.get("/measure/{image_id}")
async def measure(image_id: str):
    photo_p = MASK_DIR / f"{image_id}.png"
    mask_p  = MASK_DIR / f"{image_id}_mask.png"

    if not photo_p.exists() or not mask_p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Files not found for ID '{image_id}'. "
                   "Run /process_mobile first."
        )

    try:
        dims   = _metric_dims(photo_p, mask_p)     # width / height
        dist_m = _metric_distance(photo_p, mask_p) # wall distance
        return {**dims, "distance_m": dist_m}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
