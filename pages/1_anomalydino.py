"""Inspect one component with a selected AnomalyDINO reference bank."""

import hashlib

import streamlit as st

from inference import inspect, load_config
from ui import category_name, section_intro, show_result


category = st.session_state["selected_category"]
section_intro("Metode berbasis referensi", "AnomalyDINO", f"Pemeriksaan {category_name(category)} dengan membandingkan fitur gambar terhadap contoh komponen normal.")
scenario = st.selectbox("Jumlah referensi normal", ("AnomalyDINO 1-shot", "AnomalyDINO 16-shot", "AnomalyDINO full-shot"))
references = load_config()["categories"][category]["models"][scenario]["references"]
st.caption(f"{len(references)} gambar referensi normal · DINOv2 ViT-S/14 · batas keputusan khusus {category} × {scenario}.")
st.markdown('<div class="qc-section">Foto komponen</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Unggah gambar komponen", type=("png", "jpg", "jpeg"), key="dino_upload")

if st.button("Periksa komponen", type="primary", disabled=uploaded is None):
    image_bytes = uploaded.getvalue()
    with st.spinner("Menganalisis gambar..."):
        try:
            result = inspect(image_bytes, category, scenario)
            st.session_state["dino_result"] = (hashlib.sha256(image_bytes).hexdigest(), category, scenario, result)
        except Exception as error:
            st.error(f"Inferensi gagal: {error}")

if uploaded is not None and "dino_result" in st.session_state:
    image_bytes = uploaded.getvalue()
    fingerprint, saved_category, saved_scenario, result = st.session_state["dino_result"]
    if (fingerprint, saved_category, saved_scenario) == (hashlib.sha256(image_bytes).hexdigest(), category, scenario):
        show_result(result, image_bytes)
