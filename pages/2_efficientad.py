"""Inspect one component with the EfficientAD checkpoint for its category."""

import hashlib

import streamlit as st

from inference import inspect, load_config
from ui import category_name, section_intro, show_result


category = st.session_state["selected_category"]
section_intro("Metode model terlatih", "EfficientAD", f"Pemeriksaan {category_name(category)} menggunakan checkpoint model yang dilatih untuk kategori ini.")
st.caption(f"Checkpoint 60 epoch · batas keputusan khusus {category} × EfficientAD.")
st.markdown('<div class="qc-section">Foto komponen</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Unggah gambar komponen", type=("png", "jpg", "jpeg"), key="efficientad_upload")

if st.button("Periksa komponen", type="primary", disabled=uploaded is None):
    image_bytes = uploaded.getvalue()
    with st.spinner("Menganalisis gambar..."):
        try:
            result = inspect(image_bytes, category, "EfficientAD")
            st.session_state["efficientad_result"] = (hashlib.sha256(image_bytes).hexdigest(), category, result)
        except Exception as error:
            st.error(f"Inferensi gagal: {error}")

if uploaded is not None and "efficientad_result" in st.session_state:
    image_bytes = uploaded.getvalue()
    fingerprint, saved_category, result = st.session_state["efficientad_result"]
    if (fingerprint, saved_category) == (hashlib.sha256(image_bytes).hexdigest(), category):
        show_result(result, image_bytes)
