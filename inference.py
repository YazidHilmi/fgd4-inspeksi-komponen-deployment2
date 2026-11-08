"""Load the saved FGD artifacts and run the same inference steps as the notebook."""

import io
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from quality_guard import decode_rgb, evaluate_category_guard


ROOT = Path(__file__).resolve().parent
ANOMALYDINO_SOURCE = ROOT / "vendor" / "AnomalyDINO"
DINOV2_SOURCE = ROOT / "vendor" / "dinov2"
for source in (ANOMALYDINO_SOURCE, DINOV2_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from dinov2.hub.backbones import dinov2_vits14
from src.backbones import DINOv2Wrapper
from src.post_eval import mean_top1p
from src.utils import dists2map


MODEL_NAMES = ("AnomalyDINO 1-shot", "AnomalyDINO 16-shot", "AnomalyDINO full-shot", "EfficientAD")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data(show_spinner=False)
def load_config():
    return json.loads((ROOT / "deployment_config.json").read_text(encoding="utf-8"))


def artifact_path(relative_path):
    """Resolve a saved path inside Deployment2 and reject Git LFS pointer files."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Path artefak harus relatif terhadap Deployment2: {relative_path}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise FileNotFoundError(f"Artefak tidak ditemukan: {relative_path}")
    with resolved.open("rb") as handle:
        if handle.read(43).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Artefak masih berupa pointer Git LFS: {relative_path}")
    return resolved


class SavedDINOv2(DINOv2Wrapper):
    """Use AnomalyDINO's preprocessing and feature extraction with local weights."""

    def load_model(self):
        model = dinov2_vits14(pretrained=False)
        weights = torch.load(artifact_path(load_config()["backbone_weights"]), map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return model.eval().to(self.device)


@st.cache_resource(show_spinner=False)
def load_dino():
    config = load_config()
    if config["dino_backbone"] != "dinov2_vits14":
        raise ValueError("Backbone DINOv2 pada konfigurasi belum didukung aplikasi ini.")
    return SavedDINOv2("dinov2_vits14", DEVICE, smaller_edge_size=config["dino_input_size"])


@st.cache_resource(max_entries=1, show_spinner=False)
def load_bank(category, name):
    relative_path = load_config()["categories"][category]["models"][name]["bank"]
    index = faiss.read_index(str(artifact_path(relative_path)))
    if index.d != load_dino().model.embed_dim:
        raise ValueError(f"Dimensi memory bank tidak cocok: {category} | {name}")
    return index


@st.cache_resource(show_spinner=False)
def load_efficientad(category):
    from anomalib.models import EfficientAd

    checkpoint_path = artifact_path(load_config()["categories"][category]["efficientad_checkpoint"])
    model = EfficientAd(post_processor=False, evaluator=False, visualizer=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    optional = ("post_processor.", "evaluator.", "visualizer.")
    missing = [key for key in result.missing_keys if not key.startswith(optional)]
    unexpected = [key for key in result.unexpected_keys if not key.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint EfficientAD tidak cocok: missing={missing}, unexpected={unexpected}")
    return model.to(DEVICE).eval()


def _synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _decode_rgb(image_bytes):
    return decode_rgb(image_bytes)


def predict_dino(image_bytes, category, name):
    config = load_config()
    wrapper = load_dino()
    bank = load_bank(category, name)
    _synchronize()
    started = time.perf_counter()
    image = _decode_rgb(image_bytes)
    tensor, grid = wrapper.prepare_image(image)
    patches = wrapper.extract_features(tensor)
    foreground = wrapper.compute_background_mask(patches, grid, threshold=10, masking_type=config["categories"][category]["masking"])
    if not foreground.any():
        raise RuntimeError("Mask foreground kosong untuk gambar ini.")
    query = np.ascontiguousarray(patches[foreground], dtype=np.float32)
    faiss.normalize_L2(query)
    distances, _ = bank.search(query, 1)
    patch_scores = np.zeros(len(patches), dtype=np.float32)
    patch_scores[foreground] = distances[:, 0] / 2
    anomaly_map = dists2map(patch_scores.reshape(grid), (config["map_size"], config["map_size"])).astype(np.float32)
    _synchronize()
    return float(mean_top1p(patch_scores)), anomaly_map, time.perf_counter() - started


def predict_efficientad(image_bytes, category):
    config = load_config()
    model = load_efficientad(category)
    _synchronize()
    started = time.perf_counter()
    with Image.open(io.BytesIO(image_bytes)) as image:
        tensor = TF.to_tensor(image.convert("RGB"))
    tensor = TF.resize(tensor, [256, 256], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)[None].to(DEVICE)
    with torch.inference_mode():
        output = model.model(tensor)
    score = float(output.pred_score.detach().cpu().reshape(-1)[0])
    anomaly_map = output.anomaly_map.detach().float()
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[:, None]
    anomaly_map = F.interpolate(anomaly_map, size=(config["map_size"], config["map_size"]), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    _synchronize()
    return score, anomaly_map, time.perf_counter() - started


def inspect(image_bytes, category, name):
    config = load_config()
    if category not in config["categories"] or name not in MODEL_NAMES:
        raise ValueError("Kategori atau model tidak dikenal.")
    score, anomaly_map, elapsed = predict_efficientad(image_bytes, category) if name == "EfficientAD" else predict_dino(image_bytes, category, name)
    limits = config["categories"][category]["models"][name]["threshold"]
    guard_config = config.get("category_guard", {})
    guard = evaluate_category_guard(image_bytes, anomaly_map, limits["pixel"], score, limits["image"], area_threshold=float(guard_config.get("anomaly_area_threshold", 0.90)), enabled=bool(guard_config.get("enabled", True)))
    verdict = "CACAT" if score > limits["image"] else "NORMAL"
    return {"category": category, "model": name, "verdict": verdict, "display_verdict": "VERIFIKASI" if guard["requires_verification"] else verdict, "image_score": score, "image_threshold": limits["image"], "pixel_threshold": limits["pixel"], "anomaly_map": anomaly_map, "seconds": elapsed, **guard}
