"""
pages/prediction.py — Stress & Emotion Prediction page for MindScan.
"""

import numpy as np
import streamlit as st

from utils import clean_text


EMOTION_ICONS = {
    "happy": "😊", "sad": "😢", "anger": "😠", "fear": "😨",
    "love": "❤️", "surprise": "😲", "neutral": "😐",
}

STRESS_MAP = {
    0: ("Normal",      "😌", "pred-normal", "#22c55e"),
    1: ("Mild Stress", "😥", "pred-mild",   "#f59e0b"),
    2: ("High Stress", "😫", "pred-high",   "#ef4444"),
}


def render(slang_dict):
    st.title("🔮 Stress Prediction")

    if st.session_state.emotion_model is None or st.session_state.stress_model is None:
        st.markdown("""
        <div style='background:#1a0f0a;border:1px solid rgba(245,158,11,0.3);
                    border-radius:16px;padding:24px;text-align:center;'>
            <div style='font-size:40px;margin-bottom:12px;'>⚠️</div>
            <div style='color:#f59e0b;font-size:16px;font-weight:700;'>Model belum dilatih</div>
            <div style='color:#9999aa;font-size:13px;margin-top:8px;'>
                Silakan pergi ke halaman <b>🤖 Model Training</b> terlebih dahulu.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        user_input, predict_btn = _render_input()

    with col_result:
        if predict_btn and user_input.strip():
            _render_result(user_input, slang_dict)
        elif predict_btn:
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            st.markdown("""
            <div class="section-card" style="min-height:320px;display:flex;align-items:center;justify-content:center;text-align:center;">
                <div>
                    <div style='font-size:56px;opacity:0.25;margin-bottom:16px;'>🔮</div>
                    <div style='color:#5555aa;font-size:14px;line-height:1.7;'>
                        Masukkan teks dan klik<br>
                        <b style='color:#9090ff;'>Analisis Teks</b> untuk melihat hasil prediksi
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# INPUT PANEL
# ──────────────────────────────────────────
def _render_input():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📝 Input Teks")

    user_input = st.text_area(
        "Masukkan teks:",
        height=160,
        placeholder="Contoh: Hari ini sangat melelahkan, tugas menumpuk dan aku tidak bisa tidur...",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Contoh cepat:**")

    examples = {
        "😌 Normal":     "Hari ini sangat menyenangkan, bisa jalan-jalan dan makan enak bersama teman",
        "😥 Mild Stress":"Banyak tugas deadline minggu ini, cukup kewalahan tapi masih bisa handle",
        "😫 High Stress":"Sudah 3 hari tidak tidur karena tekanan kerja, kepala pusing dan tidak bisa konsentrasi sama sekali",
    }

    ec1, ec2, ec3 = st.columns(3)
    for col, (label, text) in zip([ec1, ec2, ec3], examples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["example_text"] = text
                st.rerun()

    if "example_text" in st.session_state and not user_input:
        user_input = st.session_state.example_text

    predict_btn = st.button("🔍 Analisis Teks", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return user_input, predict_btn


# ──────────────────────────────────────────
# RESULT PANEL
# ──────────────────────────────────────────
def _render_result(user_input, slang_dict):
    cleaned = clean_text(user_input, slang_dict)

    # Emotion prediction
    emotion_pred    = st.session_state.emotion_model.predict([cleaned])[0]
    emotion_icon    = EMOTION_ICONS.get(str(emotion_pred).lower(), "🎭")
    emotion_display = str(emotion_pred).title()

    # Stress prediction
    stress_model, tfidf = st.session_state.stress_model
    stress_vec  = tfidf.transform([cleaned])
    stress_pred = int(stress_model.predict(stress_vec)[0])
    stress_label, stress_icon, stress_cls, _ = STRESS_MAP.get(stress_pred, ("Unknown", "❓", "pred-normal", "#aaaacc"))

    # Confidence scores
    proba_html = _build_proba_html(stress_model, stress_vec)

    result_html = f"""
    <div class="section-card" style="background:linear-gradient(180deg,#111128 0%,#0d0d1f 100%);">
        <div style='font-size:13px;color:#5555aa;font-family:DM Mono;margin-bottom:18px;'>HASIL ANALISIS</div>

        <div class="pred-result" style="box-shadow:0 0 30px rgba(79,124,247,0.15);">
            <div class="pred-label">EMOSI TERDETEKSI</div>
            <div class="pred-value">{emotion_icon} {emotion_display}</div>
        </div>

        <div class="pred-result {stress_cls}" style="margin-top:14px;box-shadow:0 0 30px rgba(239,68,68,0.12);">
            <div class="pred-label">LEVEL STRES</div>
            <div class="pred-value">{stress_icon} {stress_label}</div>
        </div>

        {proba_html}

        <div style='margin-top:18px;padding:14px 16px;background:#0d0d20;border-radius:12px;border:1px solid #1e1e3f;'>
            <div style='font-size:11px;color:#5555aa;font-family:DM Mono;margin-bottom:8px;'>CLEANED TEXT</div>
            <div style='font-size:12px;color:#a0a0cc;font-family:DM Mono;line-height:1.7;'>
                {cleaned[:240]}{'...' if len(cleaned) > 240 else ''}
            </div>
        </div>
    </div>
    """
    try:
        st.html(result_html)
    except AttributeError:
        st.markdown(result_html, unsafe_allow_html=True)


def _build_proba_html(stress_model, stress_vec):
    labels_map = {
        0: ("😌 Normal",      "#22c55e"),
        1: ("😥 Mild",        "#f59e0b"),
        2: ("😫 High",        "#ef4444"),
    }
    try:
        probs = None
        if hasattr(stress_model, "predict_proba"):
            probs = stress_model.predict_proba(stress_vec)[0]
        elif hasattr(stress_model, "decision_function"):
            decision   = stress_model.decision_function(stress_vec)
            exp_scores = np.exp(decision - np.max(decision))
            probs      = (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]

        if probs is None:
            return ""

        bars = "".join(
            f"""<div style='margin:12px 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='font-size:12px;color:#c8c8ff;font-family:DM Mono;'>{label_name}</span>
                    <span style='font-size:12px;color:{color};font-family:DM Mono;font-weight:700;'>{p:.1%}</span>
                </div>
                <div style='background:#1e1e3f;border-radius:100px;height:8px;overflow:hidden;'>
                    <div style='background:{color};width:{p*100:.1f}%;height:100%;border-radius:100px;'></div>
                </div>
            </div>"""
            for i, p in enumerate(probs)
            for label_name, color in [labels_map[i]]
        )
        return f"""
        <div style='margin-top:18px;padding:18px;background:linear-gradient(135deg,#0d0d20,#13132a);
                    border-radius:16px;border:1px solid #2a2a5a;'>
            <div style='font-size:11px;color:#7777cc;font-family:DM Mono;margin-bottom:14px;'>CONFIDENCE SCORES</div>
            {bars}
        </div>
        """
    except Exception:
        return "<div style='margin-top:16px;padding:12px;border-radius:12px;background:#1a0f0f;border:1px solid rgba(239,68,68,0.3);color:#f87171;font-size:12px;'>Confidence score tidak tersedia</div>"
