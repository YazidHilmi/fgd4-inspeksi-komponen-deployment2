"""Inspect one component with a selected AnomalyDINO reference bank."""

import hashlib

import streamlit as st

from inference import inspect, load_config
from ui import category_name, show_result


category = st.session_state["selected_category"]
st.header("AnomalyDINO")
st.write(f"Inspeksi **{category_name(category)}** menggunakan fitur DINOv2 dan contoh komponen normal yang tersimpan dalam memory bank.")
scenario = st.selectbox("Jumlah referensi normal", ("AnomalyDINO 1-shot", "AnomalyDINO 16-shot", "AnomalyDINO full-shot"))
references = load_config()["categories"][category]["models"][scenario]["references"]
st.caption(f"{len(references)} gambar referensi · backbone DINOv2 ViT-S/14 · keputusan memakai threshold {category} × {scenario}.")
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
