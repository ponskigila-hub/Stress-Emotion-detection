"""
app.py — MindScan entry point.
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="MindScan · Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from styles import inject_styles, inject_sidebar_styles, set_matplotlib_theme
from data_loader import (
    load_slang_dict, load_and_preprocess,
    STRESS_LABEL_MAP, STRESS_COLORS,
    EMOTION_LABEL_MAP, EMOTION_COLORS,
)
import views.home          as page_home
import views.eda           as page_eda
import views.preprocessing as page_preprocessing
import views.training      as page_training
import views.prediction    as page_prediction

# ── Styles & theme ───────────────────────────────────────────────────────────
inject_styles()
set_matplotlib_theme()

# ── Session state ────────────────────────────────────────────────────────────
for key in ["emotion_model", "stress_model", "last_metrics", "train_log"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠  Home"

# ── Load & preprocess data (cached — runs only once) ─────────────────────────
DATA_LOADED = False
DATA_ERROR  = ""

try:
    SLANG_DICT            = load_slang_dict()
    SLANG_WORDS           = set(SLANG_DICT.keys())
    emotion_df, stress_df = load_and_preprocess(SLANG_DICT)
    DATA_LOADED           = True
except Exception as e:
    DATA_ERROR = str(e)

# ── Horizontal top navbar ────────────────────────────────────────────────────
NAV_ITEMS = [
    "🏠  Home",
    "📊  EDA",
    "⚙️  Preprocessing",
    "🤖  Model Training",
    "🔮  Prediction",
]

# Brand + nav buttons in one row
brand_col, *btn_cols, spacer = st.columns([1.8] + [1] * len(NAV_ITEMS) + [1.5])
with brand_col:
    st.markdown(
        "<div style='padding:6px 0 2px 4px;font-size:1.15rem;font-weight:800;"
        "background:linear-gradient(135deg,#fff,#6c5ce7);-webkit-background-clip:text;"
        "background-clip:text;color:transparent;'>🧠 MindScan</div>",
        unsafe_allow_html=True,
    )

for col, label in zip(btn_cols, NAV_ITEMS):
    with col:
        is_active = st.session_state.current_page == label
        if st.button(
            label,
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = label
            st.rerun()

st.markdown('<div class="topnav-divider"></div>', unsafe_allow_html=True)
page = st.session_state.current_page

# ── Sidebar — dataset status (collapsed by default) ──────────────────────────
with st.sidebar:
    inject_sidebar_styles()
    if DATA_LOADED:
        st.markdown(
            "<div style='font-size:11px;color:#5555aa;font-family:DM Mono;"
            "margin-bottom:10px;'>DATASET STATUS</div>",
            unsafe_allow_html=True,
        )
        for lv, count in stress_df["stress_label"].value_counts().sort_index().items():
            lv = int(lv)
            c  = STRESS_COLORS.get(lv, "#aaaacc")
            n  = STRESS_LABEL_MAP.get(lv, str(lv))
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"background:#13132a;border-radius:8px;padding:7px 10px;margin:4px 0;"
                f"border-left:3px solid {c};'>"
                f"<span style='font-size:11px;color:#9999cc;'>{n}</span>"
                f"<span style='font-size:12px;font-weight:700;color:{c};"
                f"font-family:DM Mono;'>{count:,}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#5555aa;font-family:DM Mono;"
            "margin-bottom:10px;'>EMOTION STATUS</div>",
            unsafe_allow_html=True,
        )
        for lv, count in emotion_df["label"].value_counts().items():
            lv_key = str(lv).lower().strip()
            c = EMOTION_COLORS.get(lv_key, "#aaaacc")
            n = EMOTION_LABEL_MAP.get(lv_key, str(lv).capitalize())
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"background:#13132a;border-radius:8px;padding:7px 10px;margin:4px 0;"
                f"border-left:3px solid {c};'>"
                f"<span style='font-size:11px;color:#9999cc;'>{n}</span>"
                f"<span style='font-size:12px;font-weight:700;color:{c};"
                f"font-family:DM Mono;'>{count:,}</span></div>",
                unsafe_allow_html=True,
            )

# ── Routing ───────────────────────────────────────────────────────────────────
if not DATA_LOADED:
    st.error(f"❌ Gagal memuat data: {DATA_ERROR}")
    st.stop()

if page == "🏠  Home":
    page_home.render(emotion_df, stress_df)
elif page == "📊  EDA":
    page_eda.render(emotion_df, stress_df, SLANG_WORDS)
elif page == "⚙️  Preprocessing":
    page_preprocessing.render(emotion_df, stress_df, SLANG_DICT)
elif page == "🤖  Model Training":
    page_training.render(emotion_df, stress_df)
elif page == "🔮  Prediction":
    page_prediction.render(SLANG_DICT)
