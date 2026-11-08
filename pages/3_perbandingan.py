"""Compare the saved models on one uploaded component image."""

import hashlib

import pandas as pd
import streamlit as st

from inference import MODEL_NAMES, inspect
from ui import category_name, show_visuals


category = st.session_state["selected_category"]
st.header("Perbandingan Model")
st.write(f"Unggah satu gambar **{category_name(category)}** untuk melihat hasil keempat kandidat pada input yang sama.")
st.caption("Setiap kandidat memakai threshold kategori dan modelnya sendiri. Pilihan model tidak digabung menjadi satu keputusan pabrik.")

# Menampilkan kemampuan tiap skenario tanpa mengklaim peringkat evaluasi yang belum tersedia.
st.subheader("Pilihan model")
st.table(pd.DataFrame([
    {"Model": "AnomalyDINO 1-shot", "Referensi normal": "1 gambar", "Cara kerja": "Membandingkan patch dengan memory bank"},
    {"Model": "AnomalyDINO 16-shot", "Referensi normal": "16 gambar", "Cara kerja": "Membandingkan patch dengan memory bank"},
    {"Model": "AnomalyDINO full-shot", "Referensi normal": "Seluruh train normal", "Cara kerja": "Membandingkan patch dengan memory bank"},
    {"Model": "EfficientAD", "Referensi normal": "Model terlatih", "Cara kerja": "Prediksi student dan autoencoder"},
]).set_index("Model"))
st.info("Tabel metrik test dari run yang menghasilkan checkpoint Deployment2 belum disertakan. Halaman ini membandingkan prediksi langsung tanpa menunjuk pemenang berdasarkan angka dari run lain.")

st.divider()
st.subheader("Coba satu gambar pada semua kandidat")
uploaded = st.file_uploader("Unggah gambar komponen", type=("png", "jpg", "jpeg"), key="comparison_upload")

# Menjalankan setiap model pada gambar dan kategori yang sama.
if st.button("Jalankan perbandingan", type="primary", disabled=uploaded is None):
    image_bytes = uploaded.getvalue()
    results = {}
    for name in ("EfficientAD", "AnomalyDINO 1-shot", "AnomalyDINO 16-shot", "AnomalyDINO full-shot"):
        with st.spinner(f"Menjalankan {name}..."):
            try:
                results[name] = inspect(image_bytes, category, name)
            except Exception as error:
                st.error(f"{name} gagal: {error}")
    st.session_state["comparison_results"] = (hashlib.sha256(image_bytes).hexdigest(), category, results)

# Menampilkan skor dan visualisasi tiap kandidat secara terpisah.
if uploaded is not None and "comparison_results" in st.session_state:
    image_bytes = uploaded.getvalue()
    fingerprint, saved_category, results = st.session_state["comparison_results"]
    if (fingerprint, saved_category) == (hashlib.sha256(image_bytes).hexdigest(), category) and results:
        rows = [{"Model": name, "Prediksi": result["verdict"], "Skor gambar": result["image_score"], "Batas keputusan": result["image_threshold"], "Waktu proses (ms)": result["seconds"] * 1000} for name, result in results.items()]
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format({"Skor gambar": "{:.5f}", "Batas keputusan": "{:.5f}", "Waktu proses (ms)": "{:.0f}"}), width="stretch")
        for name in MODEL_NAMES:
            if name in results:
                with st.expander(f"{name} · {results[name]['verdict']}", expanded=name == "EfficientAD"):
                    show_visuals(image_bytes, results[name])
        st.caption("Skor mentah antar kandidat memiliki skala berbeda. Waktu di server CPU dapat berbeda dari waktu GPU pada notebook.")
