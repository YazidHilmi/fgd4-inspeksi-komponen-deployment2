"""Shared presentation components for the three Streamlit pages."""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from inference import load_config


CATEGORY_LABELS = {"screw": "Sekrup (screw)", "metal_nut": "Mur (metal_nut)"}


def category_name(category):
    return CATEGORY_LABELS.get(category, category)


def section_intro(kicker, title, description):
    st.markdown(f'<div class="qc-section">{kicker}</div>', unsafe_allow_html=True)
    st.header(title)
    st.markdown(f'<p class="qc-lead">{description}</p>', unsafe_allow_html=True)


def show_result(result, image_bytes):
    verdict = result["verdict"]
    if verdict == "CACAT":
        st.markdown('<div class="qc-verdict bad"><strong>● CACAT TERDETEKSI · TAHAN KOMPONEN</strong><p>Indikasi cacat melewati batas keputusan. Pisahkan komponen untuk pemeriksaan petugas QC sebelum diproses lebih lanjut.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="qc-verdict good"><strong>● NORMAL · LANJUTKAN KOMPONEN</strong><p>Skor gambar berada dalam batas normal model. Komponen dapat melanjutkan ke tahap produksi berikutnya.</p></div>', unsafe_allow_html=True)

    score_col, limit_col, time_col = st.columns(3)
    score_col.metric("Skor gambar", f"{result['image_score']:.5f}")
    limit_col.metric("Batas keputusan", f"{result['image_threshold']:.5f}")
    time_col.metric("Waktu proses", f"{result['seconds'] * 1000:.0f} ms")
    st.caption("Keputusan gambar: skor gambar > batas keputusan. Area yang ditandai memakai batas piksel model terpilih; warna heatmap sendiri menunjukkan skor relatif pada gambar.")
    st.markdown('<div class="qc-section">Bukti visual</div>', unsafe_allow_html=True)
    show_visuals(image_bytes, result)


def show_visuals(image_bytes, result):
    """Match the notebook's jet heatmap and thresholded autumn overlay."""
    size = load_config()["map_size"]
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB").resize((size, size)))
    anomaly_map = result["anomaly_map"]
    pixel_limit = result["pixel_threshold"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), squeeze=False)
    panels = axes[0]
    panels[0].imshow(rgb)
    panels[1].imshow(anomaly_map, cmap="jet")
    panels[2].imshow(rgb)
    panels[2].imshow(np.ma.masked_less_equal(anomaly_map, pixel_limit), cmap="autumn", alpha=0.55)
    panels[3].imshow(anomaly_map > pixel_limit, cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(panels, ("Citra input", "Heatmap", "Area yang ditandai", "Mask prediksi")):
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    components.html(f'<img src="data:image/png;base64,{encoded}" alt="Hasil inspeksi" style="display:block;width:100%;height:auto">', height=520, scrolling=False)
