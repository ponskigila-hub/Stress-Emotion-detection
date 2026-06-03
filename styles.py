"""
styles.py — All CSS for MindScan. No logic here.
"""

import streamlit as st
import matplotlib.pyplot as plt

MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

:root {
    --ink: #e8e8f4; --ink2: #c0c0d8; --muted: #8888aa;
    --accent: #6c5ce7; --accent2: #00b894;
    --surface: #0f0f1a; --card: #1a1a2e;
    --border: rgba(108,92,231,0.2);
}

.main { background-color: var(--surface); color: var(--ink); }
.block-container { padding: 0.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── TOP NAVBAR ─────────────────────────────────────────── */
.topnav-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #13132a 0%, #1a1a35 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px 16px;
    margin-bottom: 20px;
}
.topnav-brand {
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #6c5ce7);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    white-space: nowrap;
    padding-right: 12px;
    border-right: 1px solid #2a2a5a;
    margin-right: 4px;
}
.topnav-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(108,92,231,0.3), transparent);
    margin: 0 0 18px 0;
}

/* Nav button overrides — applied only inside .stHorizontalBlock */
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 10px !important;
    padding: 7px 12px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #2a2a5a !important;
    color: #8888aa !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {
    background: rgba(108,92,231,0.1) !important;
    border-color: var(--accent) !important;
    color: #c8c8ff !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(108,92,231,0.3) !important;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%);
    border-right: 1px solid var(--border);
}

/* ── HEADINGS ────────────────────────────────────────────── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--ink) !important;
    letter-spacing: -0.5px;
    font-weight: 700;
}

/* ── BENTO CARDS ─────────────────────────────────────────── */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin: 1rem 0;
}
.bento-card {
    background: linear-gradient(135deg, #13132a 0%, #1a1a35 100%);
    border: 1px solid #2a2a5a;
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.bento-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
    border-radius: 18px 18px 0 0;
}
.bento-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.bento-label {
    font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 8px; font-family: 'DM Mono', monospace;
}
.bento-value { font-size: 2.4rem; font-weight: 800; color: #fff; line-height: 1; }
.bento-sub { font-size: 11px; color: #5a5a8a; margin-top: 6px; font-family: 'DM Mono', monospace; }

/* ── HERO BANNER ─────────────────────────────────────────── */
.hero-banner {
    background: radial-gradient(ellipse 80% 60% at 70% 50%, rgba(108,92,231,0.12) 0%, transparent 70%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 32px 36px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '🧠';
    position: absolute; right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 80px; opacity: 0.08;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #fff, var(--accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin: 0 0 8px 0;
}
.hero-sub  { color: var(--muted); font-size: 1rem; line-height: 1.5; }
.hero-pill {
    display: inline-block;
    background: rgba(108,92,231,0.2); border: 1px solid rgba(108,92,231,0.4);
    color: var(--accent2); padding: 4px 14px; border-radius: 100px;
    font-size: 12px; margin-bottom: 16px; font-family: 'DM Mono', monospace;
}

/* ── SECTION CARDS ───────────────────────────────────────── */
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 24px;
    margin: 16px 0;
}

/* ── PREDICTION PANELS ───────────────────────────────────── */
.pred-result {
    border-radius: 20px; padding: 24px 20px; text-align: center;
    margin: 12px 0; border: 1px solid transparent;
    background: linear-gradient(135deg, #0f1a40, #1a2a60);
}
.pred-normal { background: linear-gradient(135deg, #0a2a1a, #0f3020); border-color: #22c55e; }
.pred-mild   { background: linear-gradient(135deg, #2a1f0a, #3a2a0f); border-color: #f59e0b; }
.pred-high   { background: linear-gradient(135deg, #2a0a0a, #3a1010); border-color: #ef4444; }
.pred-label  {
    font-size: 12px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 10px; font-family: 'DM Mono', monospace;
}
.pred-value  { font-size: 1.8rem; font-weight: 800; color: #fff; }

/* ── DEFAULT BUTTONS ─────────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c5ce7 0%, #a855f7 100%) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 12px 22px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.25);
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 30px rgba(108,92,231,0.45) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── INPUTS ──────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextArea > div > div,
.stTextInput > div > div {
    background-color: #13132a !important;
    border: 1px solid #2a2a5a !important;
    color: var(--ink) !important;
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace;
}
.stSelectbox label, .stTextArea label, .stTextInput label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}

/* ── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d1f; border-radius: 12px;
    padding: 4px; gap: 4px; border: 1px solid #1e1e40;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: var(--muted);
    font-family: 'Syne', sans-serif; font-weight: 600; font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: white !important;
}

/* ── MISC ────────────────────────────────────────────────── */
.step-indicator {
    display: flex; align-items: center; gap: 10px; margin: 8px 0;
    color: var(--muted); font-family: 'DM Mono', monospace; font-size: 12px;
}
.step-done   { color: var(--accent2); }
.step-active { color: var(--accent); }

.balance-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(108,92,231,0.12); border: 1px solid rgba(108,92,231,0.3);
    color: var(--accent); padding: 4px 12px; border-radius: 100px;
    font-size: 11px; font-family: 'DM Mono', monospace; margin: 4px 4px 0 0;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d0d18; }
::-webkit-scrollbar-thumb { background: #2a2a5a; border-radius: 3px; }
hr { border-color: var(--border); }
</style>
"""

METRIC_CSS = """
<style>
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#13132a,#1a1a35) !important;
    border: 1px solid #2a2a5a !important;
    border-radius: 18px !important;
    padding: 22px 20px !important;
    text-align: center !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: #6c5ce7;
}
[data-testid="stMetricLabel"] {
    color: #8888cc !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: 2px !important;
    font-family: 'DM Mono', monospace !important; margin-bottom: 10px !important;
}
[data-testid="stMetricValue"] {
    color: #fff !important; font-size: 3rem !important;
    font-weight: 800 !important; line-height: 1 !important;
    margin: 0 !important; padding: 0 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }
.report-row { margin-bottom: 2px !important; }
</style>
"""


def inject_styles():
    st.markdown(MAIN_CSS,   unsafe_allow_html=True)
    st.markdown(METRIC_CSS, unsafe_allow_html=True)


def inject_sidebar_styles():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%);
        border-right: 1px solid rgba(108,92,231,0.2);
    }
    </style>
    """, unsafe_allow_html=True)


def set_matplotlib_theme():
    plt.rcParams.update({
        "axes.facecolor":   "#111128",
        "figure.facecolor": "#111128",
        "axes.edgecolor":   "#2a2a5a",
        "axes.labelcolor":  "#8888bb",
        "xtick.color":      "#7878aa",
        "ytick.color":      "#7878aa",
        "text.color":       "#ccccee",
        "grid.color":       "#1e1e3f",
        "grid.alpha":       0.5,
    })
