"""FGD Kelompok 4 — visual quality control for screws and metal nuts."""

import streamlit as st

from ui import CATEGORY_LABELS


st.set_page_config(page_title="Inspeksi Visual Komponen", page_icon="🔩", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px;}
    h1, h2, h3 {letter-spacing: -0.035em; color: #162d3d;}
    h2 {margin-top: 0.55rem;}
    [data-testid="stSidebar"] {background: #142b3a; border-right: 1px solid #314b5b;}
    [data-testid="stSidebar"] * {color: #f1f5f7;}
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {background: #263f4e;}
    [data-testid="stSidebarNav"] a {border-radius: 9px; margin: 3px 10px;}
    [data-testid="stSidebarNav"] a[aria-current="page"] {background: #315064;}
    [data-testid="stMetric"] {background: #fff; border: 1px solid #d8e2e8; border-radius: 14px; padding: 16px 18px; box-shadow: 0 4px 18px rgba(19,44,59,.045);}
    [data-testid="stMetricLabel"] {color: #526778;}
    [data-testid="stMetricValue"] {color: #172e3e; font-weight: 700;}
    [data-testid="stFileUploader"] {background: #fff; border: 1px solid #d8e2e8; border-radius: 14px; padding: 14px;}
    [data-testid="stButton"] button[kind="primary"] {background: #d96b27; border-color: #d96b27; border-radius: 9px; font-weight: 700; min-height: 44px;}
    [data-testid="stButton"] button[kind="primary"]:hover {background: #b8521e; border-color: #b8521e;}
    .qc-hero {background: linear-gradient(112deg, #122b3a 0%, #25485b 72%, #315c66 100%); color: #fff; padding: 30px 34px; border-radius: 20px; position: relative; overflow: hidden; box-shadow: 0 14px 34px rgba(22,48,62,.13);}
    .qc-hero:after {content: ""; position: absolute; width: 270px; height: 270px; border: 1px solid rgba(255,255,255,.13); border-radius: 50%; right: -50px; top: -125px; box-shadow: 0 0 0 54px rgba(255,255,255,.035), 0 0 0 108px rgba(255,255,255,.025);}
    .qc-eyebrow {font-size: .76rem; font-weight: 800; letter-spacing: .17em; text-transform: uppercase; color: #ffc494; margin-bottom: 10px;}
    .qc-hero h1 {color: #fff; font-size: clamp(1.75rem, 3vw, 2.7rem); line-height: 1.15; margin: 0 0 10px;}
    .qc-hero p {color: #d7e4e8; margin: 0; max-width: 700px; font-size: 1rem;}
    .qc-steps {display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0 26px;}
    .qc-step {background: #e8eef2; color: #355263; padding: 8px 13px; border-radius: 999px; font-size: .82rem; font-weight: 700;}
    .qc-step strong {color: #cf641f; margin-right: 6px;}
    .qc-section {color: #bc5b20; font-size: .74rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin: 20px 0 4px;}
    .qc-lead {color: #587080; margin: -4px 0 16px;}
    .qc-verdict {border-radius: 15px; padding: 20px 24px; margin: 15px 0 18px; border: 1px solid;}
    .qc-verdict strong {font-size: 1.1rem; letter-spacing: -.02em;}
    .qc-verdict p {margin: 7px 0 0; color: #395160;}
    .qc-verdict.good {background: #e7f6ee; border-color: #aedbc2; color: #166b43;}
    .qc-verdict.bad {background: #fff0ec; border-color: #efb9a7; color: #ad3b27;}
    .qc-verdict.verify {background: #fff7df; border-color: #e7c76f; color: #8a5a00;}
    .qc-note {background: #fff; border-left: 4px solid #d96b27; border-radius: 8px; padding: 13px 16px; color: #49616f; margin: 12px 0 20px;}
    .qc-sidebar-brand {font-weight: 900; font-size: 1.1rem; letter-spacing: .03em; margin: 4px 0 2px;}
    .qc-sidebar-sub {font-size: .79rem; color: #b7ccd4; margin-bottom: 24px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="qc-hero"><div class="qc-eyebrow">FGD Kelompok 4 · Stasiun QC</div><h1>Inspeksi Visual Mur &amp; Sekrup</h1><p>Periksa komponen, lihat area yang perlu ditinjau, lalu tentukan tindak lanjut pada jalur produksi.</p></div>""", unsafe_allow_html=True)
st.markdown("""<div class="qc-steps"><span class="qc-step"><strong>01</strong> Pilih komponen</span><span class="qc-step"><strong>02</strong> Unggah foto</span><span class="qc-step"><strong>03</strong> Periksa hasil</span></div>""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="qc-sidebar-brand">🔩 QC Komponen</div><div class="qc-sidebar-sub">Panel inspeksi visual · Kelompok 4</div>', unsafe_allow_html=True)
st.sidebar.selectbox("Kategori komponen", options=list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get, key="selected_category")

page = st.navigation([
    st.Page("pages/1_anomalydino.py", title="AnomalyDINO", icon=":material/search:", default=True),
    st.Page("pages/2_efficientad.py", title="EfficientAD", icon=":material/precision_manufacturing:"),
    st.Page("pages/3_perbandingan.py", title="Perbandingan", icon=":material/bar_chart:"),
])
page.run()
