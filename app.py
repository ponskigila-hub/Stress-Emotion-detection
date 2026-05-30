"""
app.py — MindScan entry point.
Run with: streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="MindScan · Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal modules ─────────────────────────────────────────────────────────
from styles import inject_styles, inject_sidebar_styles, set_matplotlib_theme
from data_loader import (
    load_data, load_slang_dict,
    STRESS_LABEL_MAP, STRESS_COLORS,
    EMOTION_LABEL_MAP, EMOTION_COLORS,
)
from utils import clean_text, detect_slang_words
import pages.home         as page_home
import pages.eda          as page_eda
import pages.preprocessing as page_preprocessing
import pages.training     as page_training
import pages.prediction   as page_prediction

# ── Global styling & matplotlib theme ───────────────────────────────────────
inject_styles()
set_matplotlib_theme()

# ── Session state ────────────────────────────────────────────────────────────
for key in ["emotion_model", "stress_model", "last_metrics", "train_log"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Load data ────────────────────────────────────────────────────────────────
DATA_LOADED = False
DATA_ERROR  = ""

try:
    emotion_df, stress_df = load_data()
    SLANG_DICT  = load_slang_dict()
    SLANG_WORDS = set(SLANG_DICT.keys())

    # Attach cleaned text & slang columns once
    emotion_df["clean_text"]  = emotion_df["text"].apply(lambda t: clean_text(t, SLANG_DICT))
    stress_df["clean_text"]   = stress_df["text"].apply(lambda t: clean_text(t, SLANG_DICT))
    emotion_df["slang_words"] = emotion_df["text"].apply(lambda t: detect_slang_words(t, SLANG_WORDS))
    emotion_df["slang_count"] = emotion_df["slang_words"].apply(len)
    stress_df["slang_words"]  = stress_df["text"].apply(lambda t: detect_slang_words(t, SLANG_WORDS))
    stress_df["slang_count"]  = stress_df["slang_words"].apply(len)

    DATA_LOADED = True
except Exception as e:
    DATA_ERROR = str(e)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    inject_sidebar_styles()

    st.markdown("""
    <div style='padding:24px 0 20px 0;text-align:center;'>
        <div style='font-size:40px;margin-bottom:8px;'>🧠</div>
        <div style='font-size:20px;font-weight:800;
                    background:linear-gradient(135deg,#fff,#6c5ce7);
                    -webkit-background-clip:text;background-clip:text;color:transparent;'>
            MindScan
        </div>
        <div style='font-size:11px;color:#5555aa;font-family:DM Mono,monospace;margin-top:4px;'>
            NLP STRESS DETECTOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Menu",
        ["🏠  Home", "📊  EDA", "⚙️  Preprocessing", "🤖  Model Training", "🔮  Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if DATA_LOADED:
        # Stress dataset status
        vc = stress_df["stress_label"].value_counts().sort_index()
        st.markdown(
            "<div style='font-size:11px;color:#5555aa;font-family:DM Mono;margin-bottom:12px;'>"
            "DATASET STATUS</div>",
            unsafe_allow_html=True,
        )
        for label_val, count in vc.items():
            lv = int(label_val)
            c  = STRESS_COLORS.get(lv, "#aaaacc")
            n  = STRESS_LABEL_MAP.get(lv, str(lv))
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        background:#13132a;border-radius:10px;padding:8px 12px;margin:6px 0;
                        border-left:3px solid {c};'>
                <span style='font-size:12px;color:#9999cc;'>{n}</span>
                <span style='font-size:13px;font-weight:700;color:{c};font-family:DM Mono;'>{count}</span>
            </div>
            """, unsafe_allow_html=True)

        # Emotion dataset status
        vc_e = emotion_df["label"].value_counts()
        st.markdown(
            "<div style='font-size:11px;color:#5555aa;font-family:DM Mono;margin-bottom:12px;'>"
            "DATASET STATUS (EMOTION)</div>",
            unsafe_allow_html=True,
        )
        for label_val, count in vc_e.items():
            lv = str(label_val).lower().strip()
            c  = EMOTION_COLORS.get(lv, "#aaaacc")
            n  = EMOTION_LABEL_MAP.get(lv, str(label_val).capitalize())
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        background:#13132a;border-radius:10px;padding:8px 12px;margin:6px 0;
                        border-left:3px solid {c};'>
                <span style='font-size:12px;color:#9999cc;'>{n}</span>
                <span style='font-size:13px;font-weight:700;color:{c};font-family:DM Mono;'>{count:,}</span>
            </div>
            """, unsafe_allow_html=True)

# ── Route to page ────────────────────────────────────────────────────────────
if not DATA_LOADED and page != "🏠  Home":
    st.error(f"❌ Gagal memuat data: {DATA_ERROR}")
    st.stop()

if page == "🏠  Home":
    if not DATA_LOADED:
        st.error(f"❌ Gagal memuat data: {DATA_ERROR}")
        st.stop()
    page_home.render(emotion_df, stress_df)

elif page == "📊  EDA":
    page_eda.render(emotion_df, stress_df, SLANG_WORDS)

elif page == "⚙️  Preprocessing":
    page_preprocessing.render(emotion_df, stress_df, SLANG_DICT)

elif page == "🤖  Model Training":
    page_training.render(emotion_df, stress_df)

elif page == "🔮  Prediction":
    page_prediction.render(SLANG_DICT)
