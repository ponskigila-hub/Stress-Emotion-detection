"""
views/prediction.py — Single-text prediction + CSV bulk analysis for MindScan.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from utils import clean_text


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
EMOTION_ICONS = {
    "happy": "😊", "sad": "😢", "sadness": "😢",
    "anger": "😠", "fear": "😨",
    "love": "❤️", "surprise": "😲", "neutral": "😐",
}

STRESS_MAP = {
    0: ("Normal",      "😌", "pred-normal", "#22c55e"),
    1: ("Mild Stress", "😥", "pred-mild",   "#f59e0b"),
    2: ("High Stress", "😫", "pred-high",   "#ef4444"),
}

STRESS_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

EMOTION_COLORS = {
    "happy": "#22c55e", "sad": "#38bdf8", "sadness": "#38bdf8",
    "anger": "#ef4444", "fear": "#a78bfa",
    "love": "#c084fc",  "surprise": "#f59e0b", "neutral": "#8888aa",
}


# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────
def render(slang_dict):
    st.title("🔮 Prediction")

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

    tab_single, tab_bulk = st.tabs([
        "   🔍  Analisis Teks Tunggal   ",
        "   👤  Analisis User Medsos   ",
    ])

    with tab_single:
        _render_single(slang_dict)

    with tab_bulk:
        _render_bulk(slang_dict)


# ─────────────────────────────────────────────────────────────
# TAB 1 — SINGLE TEXT
# ─────────────────────────────────────────────────────────────
def _render_single(slang_dict):
    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        user_input, predict_btn = _render_single_input()

    with col_result:
        if predict_btn and user_input.strip():
            _render_single_result(user_input, slang_dict)
        elif predict_btn:
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            st.markdown("""
            <div class="section-card" style="min-height:320px;display:flex;
                align-items:center;justify-content:center;text-align:center;">
                <div>
                    <div style='font-size:56px;opacity:0.25;margin-bottom:16px;'>🔮</div>
                    <div style='color:#5555aa;font-size:14px;line-height:1.7;'>
                        Masukkan teks dan klik<br>
                        <b style='color:#9090ff;'>Analisis Teks</b> untuk melihat hasil prediksi
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _render_single_input():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📝 Input Teks")

    user_input = st.text_area(
        "Masukkan teks:", height=160,
        placeholder="Contoh: Hari ini sangat melelahkan, tugas menumpuk...",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Contoh cepat:**")

    examples = {
        "😌 Normal":      "Hari ini sangat menyenangkan, bisa jalan-jalan dan makan enak bersama teman",
        "😥 Mild Stress": "Banyak tugas deadline minggu ini, cukup kewalahan tapi masih bisa handle",
        "😫 High Stress": "Sudah 3 hari tidak tidur karena tekanan kerja, kepala pusing dan tidak bisa konsentrasi sama sekali",
    }

    ec1, ec2, ec3 = st.columns(3)
    for col, (label, text) in zip([ec1, ec2, ec3], examples.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state["example_text"] = text
                st.rerun()

    if "example_text" in st.session_state and not user_input:
        user_input = st.session_state.example_text

    predict_btn = st.button("🔍 Analisis Teks", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return user_input, predict_btn


def _render_single_result(user_input, slang_dict):
    cleaned = clean_text(user_input, slang_dict)

    emotion_pred    = st.session_state.emotion_model.predict([cleaned])[0]
    emotion_icon    = EMOTION_ICONS.get(str(emotion_pred).lower(), "🎭")
    emotion_display = str(emotion_pred).title()

    stress_model, tfidf = st.session_state.stress_model
    stress_vec   = tfidf.transform([cleaned])
    stress_pred  = int(stress_model.predict(stress_vec)[0])
    stress_label, stress_icon, stress_cls, _ = STRESS_MAP.get(
        stress_pred, ("Unknown", "❓", "pred-normal", "#aaaacc")
    )

    proba_html = _build_proba_html(stress_model, stress_vec)

    result_html = f"""
    <div class="section-card" style="background:linear-gradient(180deg,#111128 0%,#0d0d1f 100%);">
        <div style='font-size:13px;color:#5555aa;font-family:DM Mono;margin-bottom:18px;'>HASIL ANALISIS</div>
        <div class="pred-result" style="box-shadow:0 0 30px rgba(79,124,247,0.15);">
            <div class="pred-label">EMOSI TERDETEKSI</div>
            <div class="pred-value">{emotion_icon} {emotion_display}</div>
        </div>
        <div class="pred-result {stress_cls}" style="margin-top:14px;">
            <div class="pred-label">LEVEL STRES</div>
            <div class="pred-value">{stress_icon} {stress_label}</div>
        </div>
        {proba_html}
        <div style='margin-top:18px;padding:14px 16px;background:#0d0d20;
                    border-radius:12px;border:1px solid #1e1e3f;'>
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


# ─────────────────────────────────────────────────────────────
# TAB 2 — BULK CSV ANALYSIS
# ─────────────────────────────────────────────────────────────
def _render_bulk(slang_dict):
    # ── Instructions ──
    st.markdown("""
    <div style='background:rgba(108,92,231,0.07);border:1px solid rgba(108,92,231,0.25);
                border-radius:16px;padding:20px 24px;margin-bottom:20px;'>
        <div style='font-size:14px;font-weight:700;color:#c8c8ff;margin-bottom:10px;'>
            📋 Cara Penggunaan — Analisis Kesehatan Mental Berbasis Riwayat Medsos
        </div>
        <div style='font-size:13px;color:#8888aa;line-height:1.8;'>
            Upload file <b style='color:#a78bfa;'>CSV</b> yang berisi kumpulan postingan/teks dari satu pengguna.<br>
            Sistem akan menganalisis setiap postingan, menghitung <b style='color:#a78bfa;'>confidence-weighted majority voting</b>,
            dan menghasilkan kesimpulan kondisi mental keseluruhan.<br><br>
            <b style='color:#c8c8ff;'>Format CSV yang dibutuhkan:</b> minimal satu kolom teks
            (nama kolom: <code style='color:#6ee7f7;'>text</code>, <code style='color:#6ee7f7;'>tweet</code>,
            <code style='color:#6ee7f7;'>post</code>, <code style='color:#6ee7f7;'>content</code>, atau kolom pertama).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Methodology explanation ──
    with st.expander("🧠 Metodologi: Confidence-Weighted Majority Voting"):
        st.markdown("""
        Berbeda dengan majority voting biasa yang hanya menghitung label terbanyak, sistem ini menggunakan **skor kepercayaan** dari setiap prediksi:

        1. **Per postingan** → model menghasilkan label + confidence score (probabilitas tiap kelas)
        2. **Akumulasi** → confidence score tiap kelas dijumlahkan di seluruh postingan
        3. **Rata-rata** → total skor dibagi jumlah postingan → rata-rata confidence per kelas
        4. **Keputusan** → kelas dengan **rata-rata confidence tertinggi** = label akhir

        Ini lebih robust karena postingan dengan kepercayaan tinggi punya bobot lebih besar daripada prediksi yang ragu-ragu.

        ```
        Contoh (3 postingan):
          Post 1 → Normal: 0.80, Mild: 0.15, High: 0.05
          Post 2 → Normal: 0.30, Mild: 0.60, High: 0.10
          Post 3 → Normal: 0.20, Mild: 0.70, High: 0.10

          Rata-rata → Normal: 0.43, Mild: 0.48, High: 0.08
          Kesimpulan → Mild Stress ✓
        ```
        """)

    # ── File upload ──
    uploaded = st.file_uploader(
        "Upload CSV postingan pengguna",
        type=["csv"],
        help="CSV dengan kolom teks postingan. Satu baris = satu postingan.",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style='border:2px dashed #2a2a5a;border-radius:16px;padding:40px;
                    text-align:center;margin-top:8px;'>
            <div style='font-size:40px;opacity:0.3;'>📂</div>
            <div style='color:#5555aa;font-size:14px;margin-top:12px;'>
                Drag & drop atau klik untuk upload CSV
            </div>
            <div style='color:#3a3a6a;font-size:12px;margin-top:6px;'>
                Format: CSV · Kolom teks wajib ada
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Parse CSV ──
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")
        return

    # Detect text column
    text_col = None
    for candidate in ["text", "tweet", "post", "content", "kalimat", "teks"]:
        if candidate in [c.lower() for c in df.columns]:
            text_col = [c for c in df.columns if c.lower() == candidate][0]
            break
    if text_col is None:
        text_col = df.columns[0]

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    if len(df) == 0:
        st.error("❌ CSV tidak memiliki baris teks yang valid.")
        return

    # Column config
    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        st.markdown(f"""
        <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:12px;
                    padding:14px 18px;margin-bottom:12px;'>
            <span style='font-size:11px;color:#6666aa;font-family:DM Mono;'>KOLOM TEKS TERDETEKSI</span><br>
            <span style='font-size:16px;font-weight:700;color:#a78bfa;'>"{text_col}"</span>
            <span style='font-size:12px;color:#6666aa;margin-left:12px;'>{len(df):,} postingan valid</span>
        </div>
        """, unsafe_allow_html=True)

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Analisis Sekarang", use_container_width=True)

    if not run_btn:
        st.markdown("##### 👀 Preview Data (5 baris pertama)")
        st.dataframe(df[[text_col]].head(5), use_container_width=True)
        return

    # ── RUN ANALYSIS ──
    _run_bulk_analysis(df, text_col, slang_dict)


# ─────────────────────────────────────────────────────────────
# BULK ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────
def _run_bulk_analysis(df, text_col, slang_dict):
    stress_model, tfidf  = st.session_state.stress_model
    emotion_model        = st.session_state.emotion_model
    n                    = len(df)

    progress_bar = st.progress(0, text="Menganalisis postingan...")

    # Accumulators for confidence-weighted voting
    stress_conf_sum  = np.zeros(3)   # sum of confidence per stress class
    emotion_conf_sum = {}            # sum of confidence per emotion class

    results = []

    for i, row in enumerate(df[text_col]):
        cleaned = clean_text(str(row), slang_dict)

        # ── Stress ──
        stress_vec  = tfidf.transform([cleaned])
        stress_pred = int(stress_model.predict(stress_vec)[0])
        stress_conf = _get_stress_proba(stress_model, stress_vec)  # array len 3
        stress_conf_sum += stress_conf

        # ── Emotion ──
        emotion_pred = str(emotion_model.predict([cleaned])[0]).lower()
        emotion_conf = _get_emotion_proba(emotion_model, cleaned)  # dict {label: score}
        for label, score in emotion_conf.items():
            emotion_conf_sum[label] = emotion_conf_sum.get(label, 0.0) + score

        results.append({
            "postingan": str(row)[:120],
            "cleaned":   cleaned[:120],
            "stress_pred":   stress_pred,
            "stress_label":  STRESS_MAP[stress_pred][0],
            "stress_conf":   float(stress_conf[stress_pred]),
            "emotion_pred":  emotion_pred,
            "emotion_conf":  float(emotion_conf.get(emotion_pred, 0.0)),
        })

        progress_bar.progress((i + 1) / n, text=f"Menganalisis {i+1}/{n} postingan...")

    progress_bar.empty()

    results_df = pd.DataFrame(results)

    # ── Compute final verdict ──
    avg_stress_conf  = stress_conf_sum / n          # avg confidence per stress class
    final_stress_idx = int(np.argmax(avg_stress_conf))
    final_stress_label, final_stress_icon, final_stress_cls, final_stress_color = STRESS_MAP[final_stress_idx]

    avg_emotion_conf  = {k: v / n for k, v in emotion_conf_sum.items()}
    final_emotion_key = max(avg_emotion_conf, key=avg_emotion_conf.get)
    final_emotion_icon = EMOTION_ICONS.get(final_emotion_key, "🎭")

    # ── VERDICT BANNER ──
    _render_verdict(
        final_stress_label, final_stress_icon, final_stress_cls, final_stress_color,
        final_emotion_key, final_emotion_icon,
        avg_stress_conf, avg_emotion_conf, n,
    )

    # ── CHARTS ──
    _render_bulk_charts(results_df, avg_stress_conf, avg_emotion_conf)

    # ── POST-LEVEL TABLE ──
    _render_post_table(results_df)

    # ── DOWNLOAD ──
    _render_download(results_df, final_stress_label, final_emotion_key)


# ─────────────────────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────────────────────
def _render_verdict(
    stress_label, stress_icon, stress_cls, stress_color,
    emotion_key, emotion_icon,
    avg_stress_conf, avg_emotion_conf, n,
):
    confidence_pct = float(np.max(avg_stress_conf)) * 100

    # Risk interpretation
    risk_map = {
        "Normal":      ("Tidak ditemukan indikasi stres signifikan berdasarkan riwayat postingan.",
                        "Pemantauan rutin disarankan.", "#22c55e"),
        "Mild Stress": ("Terdeteksi indikasi stres ringan hingga sedang.",
                        "Konsultasi lanjutan dan pemantauan lebih ketat disarankan.", "#f59e0b"),
        "High Stress": ("Terdeteksi indikasi stres tinggi yang memerlukan perhatian klinis.",
                        "Rujukan ke profesional kesehatan mental sangat disarankan.", "#ef4444"),
    }
    interp, recommend, r_color = risk_map.get(stress_label, ("", "", "#aaaacc"))

    stress_bars = "".join(
        f"""<div style='margin:8px 0;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:12px;color:#c8c8ff;font-family:DM Mono;'>
                    {STRESS_MAP[i][1]} {STRESS_MAP[i][0]}
                </span>
                <span style='font-size:12px;color:{STRESS_COLORS[i]};font-family:DM Mono;font-weight:700;'>
                    {v*100:.1f}%
                </span>
            </div>
            <div style='background:#1e1e3f;border-radius:100px;height:10px;overflow:hidden;'>
                <div style='background:{STRESS_COLORS[i]};width:{v*100:.1f}%;height:100%;border-radius:100px;'></div>
            </div>
        </div>"""
        for i, v in enumerate(avg_stress_conf)
    )

    top_emotions = sorted(avg_emotion_conf.items(), key=lambda x: x[1], reverse=True)[:4]
    emotion_bars = "".join(
        f"""<div style='margin:8px 0;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:12px;color:#c8c8ff;font-family:DM Mono;'>
                    {EMOTION_ICONS.get(k,'🎭')} {k.title()}
                </span>
                <span style='font-size:12px;color:{EMOTION_COLORS.get(k,"#aaaacc")};
                             font-family:DM Mono;font-weight:700;'>
                    {v*100:.1f}%
                </span>
            </div>
            <div style='background:#1e1e3f;border-radius:100px;height:10px;overflow:hidden;'>
                <div style='background:{EMOTION_COLORS.get(k,"#aaaacc")};width:{v*100:.1f}%;height:100%;border-radius:100px;'></div>
            </div>
        </div>"""
        for k, v in top_emotions
    )

    verdict_html = f"""
    <div style='background:linear-gradient(135deg,#0f0f1a,#1a1a30);
                border:1px solid {stress_color}44;border-radius:20px;
                padding:28px;margin:20px 0;
                box-shadow:0 0 40px {stress_color}18;'>

        <!-- Header -->
        <div style='font-size:11px;color:#5555aa;font-family:DM Mono;
                    letter-spacing:2px;margin-bottom:20px;'>
            KESIMPULAN KLINIS · {n} POSTINGAN DIANALISIS
        </div>

        <!-- Main verdict row -->
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;'>

            <div class='pred-result {stress_cls}' style='margin:0;'>
                <div class='pred-label'>TINGKAT STRES DOMINAN</div>
                <div class='pred-value'>{stress_icon} {stress_label}</div>
                <div style='font-size:12px;color:{stress_color};font-family:DM Mono;margin-top:8px;'>
                    avg confidence {confidence_pct:.1f}%
                </div>
            </div>

            <div class='pred-result' style='margin:0;'>
                <div class='pred-label'>EMOSI DOMINAN</div>
                <div class='pred-value'>{emotion_icon} {emotion_key.title()}</div>
                <div style='font-size:12px;color:#a78bfa;font-family:DM Mono;margin-top:8px;'>
                    avg confidence {avg_emotion_conf.get(emotion_key,0)*100:.1f}%
                </div>
            </div>

        </div>

        <!-- Clinical interpretation -->
        <div style='background:rgba(0,0,0,0.3);border-left:3px solid {r_color};
                    border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:20px;'>
            <div style='font-size:11px;color:#6666aa;font-family:DM Mono;margin-bottom:6px;'>
                INTERPRETASI KLINIS
            </div>
            <div style='font-size:13px;color:#ccccee;line-height:1.7;'>{interp}</div>
            <div style='font-size:12px;color:{r_color};margin-top:8px;font-weight:700;'>
                → {recommend}
            </div>
        </div>

        <!-- Confidence breakdown -->
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>
            <div>
                <div style='font-size:11px;color:#6666aa;font-family:DM Mono;margin-bottom:10px;'>
                    DISTRIBUSI STRES (AVG CONFIDENCE)
                </div>
                {stress_bars}
            </div>
            <div>
                <div style='font-size:11px;color:#6666aa;font-family:DM Mono;margin-bottom:10px;'>
                    DISTRIBUSI EMOSI (AVG CONFIDENCE)
                </div>
                {emotion_bars}
            </div>
        </div>

    </div>
    """
    try:
        st.html(verdict_html)
    except AttributeError:
        st.markdown(verdict_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
def _render_bulk_charts(results_df, avg_stress_conf, avg_emotion_conf):
    st.markdown("### 📊 Visualisasi Analisis")
    c1, c2, c3 = st.columns(3)

    # Chart 1 — stress distribution across posts
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Distribusi Stres per Postingan")
        vc = results_df["stress_label"].value_counts()
        colors = [STRESS_COLORS.get(
            next((k for k, v in {0:"Normal",1:"Mild Stress",2:"High Stress"}.items() if v==lbl), -1),
            "#aaaacc"
        ) for lbl in vc.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(vc.index, vc.values, color=colors, width=0.5, edgecolor='none')
        ax.spines[['top','right']].set_visible(False)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.3, str(val),
                    ha='center', fontsize=11, fontweight='bold', color='white')
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 2 — avg confidence per stress class
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Avg Confidence — Stres")
        labels = ["Normal", "Mild Stress", "High Stress"]
        colors2 = [STRESS_COLORS[i] for i in range(3)]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, avg_stress_conf * 100, color=colors2, width=0.5, edgecolor='none')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel("Avg Confidence (%)")
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, avg_stress_conf * 100):
            ax.text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.1f}%",
                    ha='center', fontsize=10, fontweight='bold', color='white')
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 3 — emotion distribution
    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Distribusi Emosi per Postingan")
        vc_e = results_df["emotion_pred"].value_counts()
        e_colors = [EMOTION_COLORS.get(lbl, "#aaaacc") for lbl in vc_e.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autos = ax.pie(
            vc_e.values, labels=vc_e.index,
            colors=e_colors, autopct='%1.0f%%',
            wedgeprops=dict(edgecolor='#09090f', linewidth=2),
        )
        for at in autos: at.set_color('white'); at.set_fontsize(9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 4 — confidence scatter over time
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Confidence Score Stres per Postingan (Timeline)")
    fig, ax = plt.subplots(figsize=(14, 4))
    x = range(len(results_df))
    color_list = [STRESS_COLORS.get(p, "#aaaacc") for p in results_df["stress_pred"]]
    ax.scatter(x, results_df["stress_conf"] * 100, c=color_list, s=50, alpha=0.8, zorder=3)
    ax.plot(x, results_df["stress_conf"] * 100, color="#3a3a6a", linewidth=1, alpha=0.4)
    ax.axhline(results_df["stress_conf"].mean() * 100, color="#a78bfa",
               linewidth=1.5, linestyle='--', alpha=0.7, label="Rata-rata")
    ax.set_xlabel("Postingan ke-")
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 105)
    ax.spines[['top','right']].set_visible(False)
    patches = [mpatches.Patch(color=STRESS_COLORS[i], label=STRESS_MAP[i][0]) for i in range(3)]
    patches.append(mpatches.Patch(color="#a78bfa", label="Rata-rata"))
    ax.legend(handles=patches, fontsize=9, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# POST-LEVEL TABLE
# ─────────────────────────────────────────────────────────────
def _render_post_table(results_df):
    st.markdown("### 📋 Detail Per Postingan")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    display_df = results_df[["postingan", "stress_label", "stress_conf", "emotion_pred", "emotion_conf"]].copy()
    display_df.columns = ["Postingan", "Stress Label", "Stress Conf", "Emosi", "Emotion Conf"]
    display_df["Stress Conf"] = (display_df["Stress Conf"] * 100).round(1).astype(str) + "%"
    display_df["Emotion Conf"] = (display_df["Emotion Conf"] * 100).round(1).astype(str) + "%"
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(display_df, use_container_width=True, height=350)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────
def _render_download(results_df, final_stress, final_emotion):
    st.markdown("### 💾 Export Hasil")
    col_a, col_b = st.columns(2)

    # Full CSV
    with col_a:
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Detail Lengkap (CSV)",
            data=csv_bytes,
            file_name="mindscan_bulk_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Summary CSV
    with col_b:
        summary = pd.DataFrame([{
            "total_postingan": len(results_df),
            "final_stress_verdict": final_stress,
            "final_emotion_verdict": final_emotion,
            "normal_count":     (results_df["stress_label"] == "Normal").sum(),
            "mild_count":       (results_df["stress_label"] == "Mild Stress").sum(),
            "high_count":       (results_df["stress_label"] == "High Stress").sum(),
            "avg_stress_conf":  f"{results_df['stress_conf'].mean()*100:.1f}%",
            "avg_emotion_conf": f"{results_df['emotion_conf'].mean()*100:.1f}%",
        }])
        st.download_button(
            label="⬇️ Download Ringkasan Klinis (CSV)",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="mindscan_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
# HELPERS — PROBABILITY EXTRACTION
# ─────────────────────────────────────────────────────────────
def _get_stress_proba(stress_model, stress_vec) -> np.ndarray:
    """Return confidence array of length 3 for stress classes."""
    try:
        if hasattr(stress_model, "predict_proba"):
            return stress_model.predict_proba(stress_vec)[0]
        decision   = stress_model.decision_function(stress_vec)
        exp_scores = np.exp(decision - np.max(decision))
        return (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]
    except Exception:
        pred = int(stress_model.predict(stress_vec)[0])
        arr  = np.zeros(3); arr[pred] = 1.0
        return arr


def _get_emotion_proba(emotion_model, cleaned_text: str) -> dict:
    """Return confidence dict {label: score} for emotion classes."""
    try:
        clf = emotion_model.named_steps["clf"]
        vec = emotion_model.named_steps["tfidf"].transform([cleaned_text])
        classes = clf.classes_
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(vec)[0]
        elif hasattr(clf, "decision_function"):
            decision   = clf.decision_function(vec)
            exp_scores = np.exp(decision - np.max(decision))
            probs      = (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]
        else:
            pred = clf.predict(vec)[0]
            return {c: (1.0 if c == pred else 0.0) for c in classes}
        return {str(c): float(p) for c, p in zip(classes, probs)}
    except Exception:
        pred = str(emotion_model.predict([cleaned_text])[0])
        return {pred: 1.0}


def _build_proba_html(stress_model, stress_vec) -> str:
    labels_map = {0: ("😌 Normal","#22c55e"), 1: ("😥 Mild","#f59e0b"), 2: ("😫 High","#ef4444")}
    try:
        probs = _get_stress_proba(stress_model, stress_vec)
        bars  = "".join(
            f"""<div style='margin:12px 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='font-size:12px;color:#c8c8ff;font-family:DM Mono;'>{lname}</span>
                    <span style='font-size:12px;color:{color};font-family:DM Mono;font-weight:700;'>{p:.1%}</span>
                </div>
                <div style='background:#1e1e3f;border-radius:100px;height:8px;overflow:hidden;'>
                    <div style='background:{color};width:{p*100:.1f}%;height:100%;border-radius:100px;'></div>
                </div></div>"""
            for i, p in enumerate(probs)
            for lname, color in [labels_map[i]]
        )
        return f"""<div style='margin-top:18px;padding:18px;
                    background:linear-gradient(135deg,#0d0d20,#13132a);
                    border-radius:16px;border:1px solid #2a2a5a;'>
            <div style='font-size:11px;color:#7777cc;font-family:DM Mono;margin-bottom:14px;'>
                CONFIDENCE SCORES</div>{bars}</div>"""
    except Exception:
        return "<div style='color:#f87171;font-size:12px;margin-top:12px;'>Confidence score tidak tersedia</div>"
