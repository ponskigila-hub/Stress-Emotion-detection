"""
data_loader.py — Data loading, preprocessing, caching for MindScan.
All heavy work (Sastrawi stemming/stopwords) runs ONCE and is cached.
"""

import csv
import streamlit as st
import pandas as pd


# ──────────────────────────────────────────
# RAW LOADERS (cached separately so they can
# be reused as inputs to load_and_preprocess)
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data():
    emotion_df = pd.read_csv("data/emotion_accuracy_training.csv")
    emotion_df = emotion_df.rename(columns={"tweet": "text"})

    stress_df  = pd.read_csv("data/ugm_fess_labeled.csv")
    label_col  = [c for c in stress_df.columns if "label" in c.lower()][0]
    stress_df  = stress_df.rename(columns={"full_text": "text", label_col: "stress_label"})
    stress_df["stress_label"] = (
        pd.to_numeric(
            stress_df["stress_label"].astype(str).str.replace(";", ""),
            errors="coerce",
        )
    )
    stress_df  = stress_df.dropna(subset=["stress_label", "text"])
    stress_df["stress_label"] = stress_df["stress_label"].astype(int)
    return emotion_df, stress_df


@st.cache_data(show_spinner=False)
def load_slang_dict():
    slang_dict = {}
    try:
        with open("data/slang_indo.csv", "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    slang_dict[row[0].strip().lower()] = row[1].strip().lower()
    except Exception as e:
        st.warning(f"Gagal load slang dictionary: {e}")
    return slang_dict


# ──────────────────────────────────────────
# FULL PREPROCESSING — cached so Sastrawi
# only runs once per session
# ──────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Memuat & memproses dataset… (hanya sekali)")
def load_and_preprocess(_slang_dict: dict):
    """
    Load datasets + run full clean_text pipeline.
    Uses _slang_dict prefix so Streamlit doesn't try to hash the dict arg
    (leading underscore = skip hashing, treat as immutable input).
    """
    from utils import clean_text, detect_slang_words

    emotion_df, stress_df = load_raw_data()
    slang_words = set(_slang_dict.keys())

    # Vectorised apply is still single-threaded but we avoid re-running
    # on every Streamlit rerun thanks to cache_data.
    emotion_df["clean_text"]  = emotion_df["text"].apply(
        lambda t: clean_text(t, _slang_dict)
    )
    stress_df["clean_text"]   = stress_df["text"].apply(
        lambda t: clean_text(t, _slang_dict)
    )
    emotion_df["slang_words"] = emotion_df["text"].apply(
        lambda t: detect_slang_words(t, slang_words)
    )
    emotion_df["slang_count"] = emotion_df["slang_words"].apply(len)
    stress_df["slang_words"]  = stress_df["text"].apply(
        lambda t: detect_slang_words(t, slang_words)
    )
    stress_df["slang_count"]  = stress_df["slang_words"].apply(len)

    return emotion_df, stress_df


# ──────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────
STRESS_LABEL_MAP = {0: "Normal", 1: "Mild Stress", 2: "High Stress"}
STRESS_COLORS    = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

EMOTION_LABEL_MAP = {
    "love":    "Love 🥰",
    "fear":    "Fear 😨",
    "sadness": "Sadness 😢",
    "happy":   "Happy 😄",
    "anger":   "Anger 😡",
}
EMOTION_COLORS = {
    "love":    "#c084fc",
    "fear":    "#38bdf8",
    "sadness": "#34d399",
    "happy":   "#b45309",
    "anger":   "#f43f5e",
}
