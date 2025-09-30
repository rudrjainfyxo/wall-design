
import cv2
import numpy as np
import os
import sys
import argparse
from dlab_midas import load_midas_model, predict_depth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--skip", type=int, default=3, help="Frame skip rate")
    return parser.parse_args()

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎞️  Input video FPS: {fps}")

    midas_model, midas_transform, midas_device = load_midas_model()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip == 0:
            resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            depth_map = predict_depth(resized, midas_model, midas_transform, midas_device)

            norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            color_map = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
            combined = np.hstack((resized, color_map))

            cv2.imshow("Depth Estimation", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
