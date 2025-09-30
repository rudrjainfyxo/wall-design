#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wall_mask_working.py  ─  DeepLab V3+ (MobileNet-V2, ADE-20K) demo
• Accepts an image or a video file.
• Generates a wall mask (class-12) and overlays it in green.
• Defaults to CPUExecutionProvider; add --coreml to try CoreMLExecutionProvider.
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os, sys, time

# ───────────── constants ────────────────────────────────────────────────────
ONNX_FILE   = "TopFormer-S_512x512_2x8_160k.onnx"    # place next to this script
LABEL_WALL  = 0                             # ADE-20K index for “wall”
INPUT_SIZE  = 512                             # model expects 513×513 RGB
# ────────────────────────────────────────────────────────────────────────────


class DeepLab:
    """Minimal DeepLab loader + inference (NHWC 513×513, opset 11/12)."""

    def __init__(self, onnx_path: str, use_coreml: bool = False):
        if not os.path.isfile(onnx_path):
            sys.exit(f"❌  ONNX file not found: {onnx_path}")

        # ---------- dynamic provider list ---------------------------------
        avail = ort.get_available_providers()
        if use_coreml and "CoreMLExecutionProvider" in avail:
            prefs = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            prefs = ["CPUExecutionProvider"]

        print("🔹  ORT providers ➜", prefs)
        self.session  = ort.InferenceSession(onnx_path, providers=prefs)
        self.in_name  = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    # ---------------------------------------------------------------------
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Return uint8 mask resized to original WxH."""
        h0, w0 = bgr.shape[:2]

        # BGR ➜ RGB ➜ 513×513 ➜ float32 [-1,1] ➜ NHWC
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
        inp = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32) # (1,3,512,512)

        logits = self.session.run(
            [self.out_name], {self.in_name: inp}
        )[0]                                               # 1×150×513×513
        labels = np.argmax(logits, axis=1)[0].astype(np.uint8)
        mask   = cv2.resize(labels, (w0, h0), cv2.INTER_NEAREST)
        return mask
    # ---------------------------------------------------------------------


# ───────────── helpers ─────────────────────────────────────────────────────
def process_image(path: str, model: DeepLab):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"❌  Cannot read {path}")

    t0   = time.time()
    mask = model.infer(img)
    print(f"✓ Inference {time.time() - t0:.3f}s")

    vis = img.copy()
    vis[mask == LABEL_WALL] = (0, 255, 0)          # green overlay
    cv2.imshow("wall mask", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(path: str, out_path: str, model: DeepLab):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open {path}")

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw   = cv2.VideoWriter(out_path,
                           cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (w, h))

    frame_i, t_inf, mask = 0, 0.0, None
    reuse_n = 1                              # reuse last mask every n frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_i % reuse_n == 0:
            t0   = time.time()
            mask = model.infer(frame)
            t_inf += time.time() - t0

        frame[mask == LABEL_WALL] = (0, 255, 0)
        vw.write(frame)
        cv2.imshow("wall mask", frame)
        if cv2.waitKey(1) == 27:             # ESC quits early
            break
        frame_i += 1

    cap.release(), vw.release(), cv2.destroyAllWindows()
    if frame_i:
        print(f"✓ {frame_i} frames  |  mean inf {t_inf/frame_i:.3f}s  |  saved ➜ {out_path}")


# ───────────── main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="DeepLab wall-mask demo")
    ap.add_argument("src",  help="input image / video file")
    ap.add_argument("--out",   default="out.mp4",   help="output video path")
    ap.add_argument("--model", default=ONNX_FILE,   help="ONNX model path")
    ap.add_argument("--coreml", action="store_true",
                    help="try CoreMLExecutionProvider if available")
    args = ap.parse_args()

    model = DeepLab(args.model, use_coreml=args.coreml)

    if args.src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        process_image(args.src, model)
    else:
        process_video(args.src, args.out, model)


if __name__ == "__main__":
    main()
