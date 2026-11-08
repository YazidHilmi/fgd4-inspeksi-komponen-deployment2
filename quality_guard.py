"""Heuristic quality guard for inputs far outside the normal component profile."""

import cv2
import numpy as np


def decode_rgb(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Gambar tidak dapat dibaca. Gunakan PNG atau JPG yang valid.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def estimate_foreground_mask(image_bytes, size):
    """Estimate the component area from its difference to the image border."""
    image = cv2.resize(decode_rgb(image_bytes), (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_width = max(2, size // 32)
    border = np.concatenate((
        lab[:border_width].reshape(-1, 3),
        lab[-border_width:].reshape(-1, 3),
        lab[:, :border_width].reshape(-1, 3),
        lab[:, -border_width:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    scaled = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(3, (size // 64) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return np.ones((size, size), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    foreground = labels == largest
    if foreground.mean() < 0.005:
        return np.ones((size, size), dtype=bool)
    return foreground


def evaluate_category_guard(image_bytes, anomaly_map, pixel_threshold, image_score, image_threshold, area_threshold=0.90, enabled=True):
    """Flag inputs whose detected anomaly covers almost the whole component."""
    foreground = estimate_foreground_mask(image_bytes, anomaly_map.shape[0])
    anomaly_mask = anomaly_map > pixel_threshold
    foreground_pixels = int(foreground.sum())
    anomaly_pixels = int(np.logical_and(anomaly_mask, foreground).sum())
    anomaly_ratio = anomaly_pixels / foreground_pixels if foreground_pixels else 0.0
    requires_verification = bool(enabled and image_score > image_threshold and anomaly_ratio >= area_threshold)
    return {
        "foreground_mask": foreground,
        "anomaly_mask": anomaly_mask,
        "object_coverage": float(foreground.mean()),
        "anomaly_object_ratio": float(anomaly_ratio),
        "verification_threshold": float(area_threshold),
        "requires_verification": requires_verification,
    }
