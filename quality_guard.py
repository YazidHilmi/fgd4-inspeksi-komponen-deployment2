"""Shape-based input validation for the two supported component categories."""

import cv2
import numpy as np


def decode_rgb(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Gambar tidak dapat dibaca. Gunakan PNG atau JPG yang valid.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _largest_component(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count < 2:
        return np.zeros(binary.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def estimate_foreground_mask(image_bytes, size, category):
    """Segment the dominant component using the acquisition style of MVTec AD."""
    image = cv2.resize(decode_rgb(image_bytes), (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_mode = cv2.THRESH_BINARY_INV if category == "screw" else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, threshold_mode + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_component(binary)


def measure_object_shape(mask):
    """Return scale-independent measurements from a binary component mask."""
    height, width = mask.shape
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)
    if len(points) < 5:
        return {"area_ratio": 0.0, "major_axis_ratio": 0.0, "minor_axis_ratio": 0.0, "elongation": 0.0, "border_touch_ratio": 1.0}
    (_, _), (box_width, box_height), _ = cv2.minAreaRect(points)
    major = max(box_width, box_height)
    minor = max(min(box_width, box_height), 1e-6)
    border_pixels = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return {
        "area_ratio": float(mask.mean()),
        "major_axis_ratio": float(major / max(height, width)),
        "minor_axis_ratio": float(minor / max(height, width)),
        "elongation": float(major / minor),
        "border_touch_ratio": float(border_pixels / max(int(mask.sum()), 1)),
    }


def evaluate_input_shape_guard(image_bytes, category, guard_config, size=256):
    """Check whether the segmented object resembles the selected category's framing."""
    mask = estimate_foreground_mask(image_bytes, size, category)
    measurements = measure_object_shape(mask)
    category_limits = guard_config.get("categories", {}).get(category, {})
    reasons = []
    labels = {
        "area_ratio": "luas objek",
        "major_axis_ratio": "panjang objek",
        "minor_axis_ratio": "lebar objek",
        "elongation": "rasio bentuk objek",
    }
    for feature, label in labels.items():
        limits = category_limits.get(feature)
        if limits and not (float(limits[0]) <= measurements[feature] <= float(limits[1])):
            reasons.append(f"{label} di luar rentang {category}")
    maximum_border_touch = float(category_limits.get("maximum_border_touch_ratio", 0.02))
    if measurements["border_touch_ratio"] > maximum_border_touch:
        reasons.append("objek menyentuh tepi gambar secara tidak wajar")
    if not mask.any():
        reasons.append("objek utama tidak berhasil ditemukan")
    enabled = bool(guard_config.get("enabled", True))
    return {
        "foreground_mask": mask,
        "object_area_ratio": measurements["area_ratio"],
        "object_major_axis_ratio": measurements["major_axis_ratio"],
        "object_minor_axis_ratio": measurements["minor_axis_ratio"],
        "object_elongation": measurements["elongation"],
        "object_border_touch_ratio": measurements["border_touch_ratio"],
        "shape_guard_reasons": reasons,
        "requires_verification": bool(enabled and reasons),
    }
