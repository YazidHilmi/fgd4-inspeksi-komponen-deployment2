"""Calibrate deployment-only shape ranges from MVTec train/good images."""

import argparse
import json
from pathlib import Path

import numpy as np

from quality_guard import estimate_foreground_mask, measure_object_shape


FEATURES = ("area_ratio", "major_axis_ratio", "minor_axis_ratio", "elongation")


def calibrate_category(folder, category, size=256, margin_fraction=0.60):
    rows = []
    for image_path in sorted(folder.glob("*.png")):
        mask = estimate_foreground_mask(image_path.read_bytes(), size, category)
        rows.append(measure_object_shape(mask))
    if not rows:
        raise FileNotFoundError(f"Tidak ada PNG normal di {folder}")
    calibrated = {}
    for feature in FEATURES:
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        minimum, maximum = float(values.min()), float(values.max())
        margin = max((maximum - minimum) * margin_fraction, 0.005 if feature != "elongation" else 0.15)
        calibrated[feature] = [round(max(0.0, minimum - margin), 6), round(maximum + margin, 6)]
    border_values = np.asarray([row["border_touch_ratio"] for row in rows], dtype=np.float64)
    if category == "metal_nut":
        # Allow silhouette-changing defects while still rejecting objects that nearly fill the frame.
        calibrated["major_axis_ratio"][1] = max(calibrated["major_axis_ratio"][1], 0.95)
    calibrated["maximum_border_touch_ratio"] = round(max(0.02, float(border_values.max()) + 0.005), 6)
    calibrated["calibration_images"] = len(rows)
    return calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Folder yang berisi screw/ dan metal_nut/")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deployment_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shape_guard = {"enabled": True, "method": "min_max_train_normal_with_defect_tolerant_margin", "categories": {}}
    for category in ("screw", "metal_nut"):
        folder = args.data_root / category / "train" / "good"
        shape_guard["categories"][category] = calibrate_category(folder, category)
    config["input_shape_guard"] = shape_guard
    config.pop("category_guard", None)
    args.config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(shape_guard, indent=2))


if __name__ == "__main__":
    main()
