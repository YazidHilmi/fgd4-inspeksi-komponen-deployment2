"""FGD Kelompok 4 — three-page visual inspection application."""

import streamlit as st

from ui import CATEGORY_LABELS


st.set_page_config(page_title="Inspeksi Visual Komponen", page_icon="🔩", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; max-width: 1320px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    [data-testid="stMetric"] {background: #f3f7fb; border: 1px solid #dce6ee; border-radius: 12px; padding: 14px;}
</style>
""", unsafe_allow_html=True)

st.title("Inspeksi Visual Komponen")
st.caption("FGD Kelompok 4 · Deteksi dan lokalisasi cacat pada sekrup dan mur")
st.sidebar.selectbox("Kategori komponen", options=list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get, key="selected_category")

page = st.navigation([
    st.Page("pages/1_anomalydino.py", title="AnomalyDINO", icon=":material/search:", default=True),
    st.Page("pages/2_efficientad.py", title="EfficientAD", icon=":material/precision_manufacturing:"),
    st.Page("pages/3_perbandingan.py", title="Perbandingan", icon=":material/bar_chart:"),
])
page.run()
