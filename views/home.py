"""
pages/home.py — Home page for MindScan.
"""

import matplotlib.pyplot as plt
import streamlit as st


def render(emotion_df, stress_df):
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-pill">NLP · Machine Learning · Mental Health</div>
        <div class="hero-title">MindScan</div>
        <p class="hero-sub">
            Deteksi emosi dan tingkat stres dari teks media sosial menggunakan<br>
            Natural Language Processing dan Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    stress_dist = stress_df["stress_label"].value_counts().sort_index()

    st.markdown(f"""
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-label">Total Data Emosi</div>
            <div class="bento-value">{len(emotion_df):,}</div>
            <div class="bento-sub">{emotion_df["label"].nunique()} kelas emosi</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">Total Data Stres</div>
            <div class="bento-value">{len(stress_df):,}</div>
            <div class="bento-sub">3 level stres</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">Normal</div>
            <div class="bento-value">{stress_dist.get(0, 0):,}</div>
            <div class="bento-sub">label 0</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">Stress</div>
            <div class="bento-value">{stress_dist.get(1, 0) + stress_dist.get(2, 0):,}</div>
            <div class="bento-sub">label 1 + 2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### 🗂️ Alur Penggunaan Aplikasi")
        steps = [
            ("📊", "EDA", "Eksplorasi distribusi data dan wordcloud"),
            ("⚙️", "Preprocessing", "Lihat hasil pembersihan teks"),
            ("🤖", "Model Training", "Pilih model + strategi balancing, lalu train"),
            ("🔮", "Prediction", "Input teks untuk deteksi emosi & stres"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:14px; align-items:flex-start; padding:14px;
                        border-radius:12px; margin:8px 0; background:rgba(108,92,231,0.05);
                        border:1px solid #1e1e40;'>
                <span style='font-size:22px'>{icon}</span>
                <div>
                    <div style='font-weight:700; color:#c8c8ff; font-size:14px;'>{title}</div>
                    <div style='color:#6666aa; font-size:12px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 Distribusi Label Stres")
        labels_map = {0: "Normal", 1: "Mild Stress", 2: "High Stress"}
        bar_colors = ["#22c55e", "#f59e0b", "#ef4444"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [labels_map.get(int(k), str(k)) for k in stress_dist.index]
        ax.bar(x, stress_dist.values, color=bar_colors[:len(x)], width=0.5, edgecolor='none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel("Jumlah", color="#7878aa")
        for i, (xi, vi) in enumerate(zip(x, stress_dist.values)):
            ax.text(i, vi + 5, f"{vi:,}", ha='center', fontsize=11, fontweight='bold', color='white')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
