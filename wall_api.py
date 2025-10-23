#!/usr/bin/env python3
"""
wall_api.py – FastAPI wrapper around two mask generators
   • /process_mobile  → wall_mask_generator_mobile.py  (ViT-B)
   • /process_hq      → wall_mask_generator.py         (ViT-L HQ)
Launch:
    uvicorn wall_api:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid, os, shutil, importlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import cv2

# ── helper class that keeps models resident ──────────────────────
class WallRefiner:
    """Wraps one generator module (mobile or hq) and keeps its weights in RAM."""
    def __init__(self, rw_module):
        self.rw    = rw_module
        self.dl    = rw_module.DeepLab()
        self.yolo  = rw_module.YOLO(rw_module.YOLO_WEIGHTS)

    def run(self, img_path: str):
        rw   = self.rw
        img  = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image")

        dense   = rw.deeplab_bool(img, self.dl)
        masks   = rw.yolo_masks(img_path, self.yolo, rw.CONF_DEF, dense)
        union   = rw.central_union(masks, *img.shape[:2])
        refined = rw.hqsam_refine(img, union)
        if refined.sum() == 0:
            raise ValueError("No wall mask produced")

        quad = rw.wall_quad(refined)
        pitch, yaw, roll, normal = rw.wall_pose(quad, img.shape[1], img.shape[0])

        out_mask = Path(img_path).with_suffix(".mask.png")
        cv2.imwrite(str(out_mask), refined)

        return {
            "mask_path": str(out_mask),
            "pitch": pitch,
            "yaw":   yaw,
            "roll":  roll,
            "normal": normal,
        }

# ── load both generators once ────────────────────────────────────
gen_mobile = importlib.import_module("wall_mask_generator_mobile")  # ViT-B
gen_hq     = importlib.import_module("wall_mask_generator")         # ViT-L

refiners = {
    "mobile": WallRefiner(gen_mobile),
    "hq":     WallRefiner(gen_hq),
}

# ── FastAPI boilerplate ──────────────────────────────────────────
MASK_DIR = Path(__file__).parent / "masks"
MASK_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Wall-Mask Refinement API",
    description="YOLO ∩ DeepLab ∪ HQ-SAM (mobile & HQ) wrapped in FastAPI",
    version="0.4.0",
)

app.mount("/masks", StaticFiles(directory=str(MASK_DIR)), name="masks")

# ── shared handler ------------------------------------------------
def _handle(file: UploadFile, model_key: str):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="JPEG or PNG only")

    tmp_path = f"/tmp/{uuid.uuid4().hex}{Path(file.filename).suffix}"
    with open(tmp_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        res = refiners[model_key].run(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    out_name = f"{uuid.uuid4().hex}.png"
    shutil.move(res["mask_path"], MASK_DIR / out_name)

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
        "mask_url": f"/masks/{out_name}",
    }

# ── two explicit routes ------------------------------------------
@app.post("/process_mobile")
async def process_mobile(file: UploadFile = File(...)):
    "Runs the lightweight ViT-B SAM model."
    return _handle(file, "mobile")

@app.post("/process_hq")
async def process_hq(file: UploadFile = File(...)):
    "Runs the heavier HQ ViT-L SAM model."
    return _handle(file, "hq")
