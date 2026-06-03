"""
data_loader.py — Data loading, caching, and dataset utilities for MindScan.
"""

import csv
import streamlit as st
import pandas as pd


# ==========================================
# LOAD DATASETS
# ==========================================
@st.cache_data
def load_data():
    """Load and return emotion and stress DataFrames."""
    emotion_df = pd.read_csv("data/emotion_accuracy_training.csv")
    emotion_df = emotion_df.rename(columns={"tweet": "text"})

    stress_df = pd.read_csv("data/ugm_fess_labeled.csv")
    label_col = [c for c in stress_df.columns if "label" in c.lower()][0]
    stress_df = stress_df.rename(columns={"full_text": "text", label_col: "stress_label"})
    stress_df["stress_label"] = (
        stress_df["stress_label"].astype(str).str.replace(";", "")
    )
    stress_df["stress_label"] = pd.to_numeric(stress_df["stress_label"], errors="coerce")
    stress_df = stress_df.dropna(subset=["stress_label", "text"])
    stress_df["stress_label"] = stress_df["stress_label"].astype(int)
    return emotion_df, stress_df


# ==========================================
# LOAD SLANG DICTIONARY
# ==========================================
@st.cache_data
def load_slang_dict():
    """Load slang→formal mapping from CSV."""
    slang_dict = {}
    try:
        with open("data/slang_indo.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    slang = str(row[0]).strip().lower()
                    formal = str(row[1]).strip().lower()
                    slang_dict[slang] = formal
    except Exception as e:
        st.warning(f"Gagal load slang dictionary: {e}")
    return slang_dict


# ==========================================
# LABEL / COLOR CONSTANTS
# ==========================================
STRESS_LABEL_MAP = {0: "Normal", 1: "Mild Stress", 2: "High Stress"}
STRESS_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

EMOTION_LABEL_MAP = {
    "love": "Love 🥰",
    "fear": "Fear 😨",
    "sadness": "Sadness 😢",
    "happy": "Happy 😄",
    "anger": "Anger 😡",
}
EMOTION_COLORS = {
    "love": "#c084fc",
    "fear": "#38bdf8",
    "sadness": "#34d399",
    "happy": "#b45309",
    "anger": "#f43f5e",
}
