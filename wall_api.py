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

        if ratio < 0.2:
            print(f"[WARN] Fallback → refined mask area only {ratio*100:.1f}% of image. Using DeepLab mask.")
            refined = (dense.astype('uint8')) * 255
            res_fallback = True


        if refined.sum() == 0:
            raise ValueError("No wall mask produced")

        # ─ pose calc ─
        t4 = time.perf_counter()
        quad = self.rw.wall_quad(refined)
        pitch, yaw, roll, normal = self.rw.wall_pose(quad, img.shape[1], img.shape[0])
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
            "debug": debug
        }

# ─── Model modules ───────────────────────────────────────────────
gen_mobile = importlib.import_module("wall_mask_generator_mobile")
gen_hq     = importlib.import_module("wall_mask_generator")

refiners = {
    "mobile": WallRefiner(gen_mobile),
    "hq":     WallRefiner(gen_hq),
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

# ─── Core handler ────────────────────────────────────────────────
# ───────── core handler ─────────────────────────────────────────
def _handle(file: UploadFile, model_key: str):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="JPEG or PNG only")

    # one UUID for both original photo and generated mask
    uid = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower() or ".jpg"
    tmp = f"/tmp/{uid}{ext}"

    # save upload to temp file
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        # run full pipeline
        res = refiners[model_key].run(tmp)

        # move the original image next to masks/
        orig_name = f"{uid}{ext}"
        shutil.move(tmp, MASK_DIR / orig_name)

    except Exception as exc:
        if os.path.exists(tmp):           # cleanup on failure
            os.remove(tmp)
        raise HTTPException(status_code=500, detail=str(exc))

    # move the refined mask (always .png) next to original
    mask_name = f"{uid}.png"
    shutil.move(res["mask_path"], MASK_DIR / mask_name)

    # optional wall-size values (only if present in res)
    wall_size = (
        {"width": res["width_m"], "height": res["height_m"]}
        if "width_m" in res and "height_m" in res
        else None
    )

    return {
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
        "original_url": f"/masks/{orig_name}",
        "mask_url":     f"/masks/{mask_name}",
        "timings_s":    res["timings"],
        "debug":        res["debug"],
    }

# ─── API endpoints ───────────────────────────────────────────────
@app.post("/process_mobile")
async def process_mobile(file: UploadFile = File(...)):
    "Runs the lightweight ViT-B SAM model."
    return _handle(file, "mobile")

@app.post("/process_hq")
async def process_hq(file: UploadFile = File(...)):
    "Runs the heavier HQ ViT-L SAM model."
    return _handle(file, "hq")
