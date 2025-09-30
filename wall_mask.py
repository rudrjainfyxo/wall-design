import cv2
import numpy as np
from midas_depth import load_midas_model, estimate_depth

def create_wall_mask(depth_map, quantile=0.3):
    normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    mask = (normalized <= np.quantile(normalized, quantile)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [largest], -1, 255, -1)
        return filled
    return mask
