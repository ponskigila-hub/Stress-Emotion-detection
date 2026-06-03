"""
pages/eda.py — Exploratory Data Analysis page for MindScan.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from html import escape
from wordcloud import WordCloud

from data_loader import STRESS_LABEL_MAP


def render(emotion_df, stress_df, slang_words):
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "         Emosi        ",
        "          Stres       ",
        "          WordCloud      ",
        "          Slang Analysis      ",
    ])

    with tab1:
        _render_emotion_tab(emotion_df)

    with tab2:
        _render_stress_tab(stress_df)

    with tab3:
        _render_wordcloud_tab(emotion_df, stress_df)

    with tab4:
        _render_slang_tab(emotion_df, stress_df, slang_words)


# ──────────────────────────────────────────
# TAB 1 — EMOTION
# ──────────────────────────────────────────
def _render_emotion_tab(emotion_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Preview Data Emosi")
    st.dataframe(emotion_df[["text", "label", "clean_text"]].head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Distribusi Kelas Emosi")
    vc = emotion_df["label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    palette = sns.color_palette("husl", len(vc))

    bars = axes[0].barh(vc.index, vc.values, color=palette, edgecolor='none', height=0.6)
    axes[0].spines[['top', 'right', 'left']].set_visible(False)
    axes[0].set_xlabel("Jumlah")
    for bar, val in zip(bars, vc.values):
        axes[0].text(val + 5, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va='center', fontsize=10, color='white')

    wedges, texts, autotexts = axes[1].pie(
        vc.values, labels=vc.index, autopct='%1.1f%%',
        colors=palette, startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor='#09090f', linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('white')
    axes[1].set_facecolor('#111128')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# TAB 2 — STRESS
# ──────────────────────────────────────────
def _render_stress_tab(stress_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Preview Data Stres")
    st.dataframe(stress_df[["text", "stress_label", "clean_text"]].head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    bar_colors = ["#22c55e", "#f59e0b", "#ef4444"]

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Distribusi Label Stres (Original)")
        vc_s = stress_df["stress_label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        x_labels = [STRESS_LABEL_MAP.get(int(k), str(k)) for k in vc_s.index]
        bars = ax.bar(x_labels, vc_s.values, color=bar_colors[:len(vc_s)], width=0.5, edgecolor='none')
        ax.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars, vc_s.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:,}",
                    ha='center', fontsize=11, fontweight='bold', color='white')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Panjang Teks (Distribusi)")
        stress_df["text_len"] = stress_df["text"].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots(figsize=(5, 4))
        for lbl, color in [(0, "#22c55e"), (1, "#f59e0b"), (2, "#ef4444")]:
            subset = stress_df[stress_df["stress_label"] == lbl]["text_len"]
            if len(subset) > 0:
                ax.hist(subset, bins=30, alpha=0.6, color=color,
                        label=STRESS_LABEL_MAP.get(lbl, str(lbl)), edgecolor='none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Jumlah Kata")
        ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# TAB 3 — WORDCLOUD
# ──────────────────────────────────────────
def _render_wordcloud_tab(emotion_df, stress_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    wc_choice = st.selectbox(
        "Pilih Dataset untuk WordCloud",
        ["Semua Emosi", "Normal (label 0)", "Mild Stress (label 1)", "High Stress (label 2)"],
    )
    text_map = {
        "Semua Emosi": " ".join(emotion_df["text"].astype(str)),
        "Normal (label 0)": " ".join(stress_df[stress_df["stress_label"] == 0]["text"].astype(str)),
        "Mild Stress (label 1)": " ".join(stress_df[stress_df["stress_label"] == 1]["text"].astype(str)),
        "High Stress (label 2)": " ".join(stress_df[stress_df["stress_label"] == 2]["text"].astype(str)),
    }
    wc_colors = {
        "Semua Emosi": "viridis",
        "Normal (label 0)": "Greens",
        "Mild Stress (label 1)": "YlOrBr",
        "High Stress (label 2)": "Reds",
    }
    text = text_map[wc_choice]
    if text.strip():
        wc = WordCloud(
            width=1200, height=450, background_color="#111128",
            colormap=wc_colors[wc_choice], max_words=120, prefer_horizontal=0.85,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor('#111128')
        fig.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close()

    # Slang WordCloud
    st.markdown("---")
    st.markdown("##### ☁️ Slang WordCloud")
    slang_wc_dataset = st.selectbox(
        "Pilih Dataset Slang", ["Emotion Dataset", "Stress Dataset"], key="slang_wc"
    )
    source_df = emotion_df if slang_wc_dataset == "Emotion Dataset" else stress_df
    all_slangs = [w for words in source_df["slang_words"] for w in words]
    wc_text = " ".join(all_slangs)

    if wc_text.strip():
        wc = WordCloud(
            width=1200, height=400, background_color="#111128", colormap="plasma"
        ).generate(wc_text)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Tidak ada slang terdeteksi.")

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# TAB 4 — SLANG ANALYSIS
# ──────────────────────────────────────────
def _render_slang_tab(emotion_df, stress_df, slang_words):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 🔤 Analisis Slang Language")

    dataset_choice = st.selectbox("Pilih Dataset", ["Emotion Dataset", "Stress Dataset"])
    df_slang = emotion_df if dataset_choice == "Emotion Dataset" else stress_df

    total_slang = int(df_slang["slang_count"].sum())
    total_texts = len(df_slang)
    texts_with_slang = int((df_slang["slang_count"] > 0).sum())
    slang_pct = texts_with_slang / total_texts * 100

    bento_html = f"""
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-label">TOTAL SLANG</div>
            <div class="bento-value">{total_slang:,}</div>
            <div class="bento-sub">kata slang ditemukan</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">TEXT WITH SLANG</div>
            <div class="bento-value">{texts_with_slang:,}</div>
            <div class="bento-sub">mengandung slang</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">SLANG PERCENTAGE</div>
            <div class="bento-value">{slang_pct:.1f}%</div>
            <div class="bento-sub">dari total teks</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">SLANG VOCAB</div>
            <div class="bento-value">{len(slang_words):,}</div>
            <div class="bento-sub">kamus slang</div>
        </div>
    </div>
    """
    try:
        st.html(bento_html)
    except Exception:
        st.markdown(bento_html, unsafe_allow_html=True)

    all_slangs = [w for words in df_slang["slang_words"] for w in words]
    slang_counter = Counter(all_slangs)

    if slang_counter:
        top_slangs = slang_counter.most_common(15)
        import pandas as pd
        slang_df = pd.DataFrame(top_slangs, columns=["Slang", "Frequency"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📊 Top Slang Words")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(slang_df["Slang"][::-1], slang_df["Frequency"][::-1])
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("##### 🥧 Slang Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(slang_df["Frequency"][:8], labels=slang_df["Slang"][:8], autopct='%1.1f%%')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Before vs After
        st.markdown("##### 🔄 Before vs After Slang Normalization")
        samples = df_slang[df_slang["slang_count"] > 0].sample(
            min(8, texts_with_slang), random_state=42
        )
        for _, row in samples.iterrows():
            original = escape(str(row["text"]))
            cleaned = escape(str(row["clean_text"]))
            slang_found = ", ".join(row["slang_words"])
            card_html = f"""
            <div style='background:#13132a; border:1px solid #2a2a5a; border-radius:14px;
                        padding:16px; margin:12px 0;'>
                <div style='font-size:11px; color:#f59e0b; font-family:DM Mono; margin-bottom:6px;'>ORIGINAL</div>
                <div style='font-size:13px; color:#ccccff; margin-bottom:12px; line-height:1.6;'>{original[:250]}</div>
                <div style='font-size:11px; color:#22c55e; font-family:DM Mono; margin-bottom:6px;'>NORMALIZED</div>
                <div style='font-size:13px; color:#ffffff; margin-bottom:10px; line-height:1.6;'>{cleaned[:250]}</div>
                <div style='font-size:11px; color:#8888aa;'>
                    Slang detected: <span style='color:#a78bfa'>{slang_found}</span>
                </div>
            </div>
            """
            try:
                st.html(card_html)
            except Exception:
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada slang terdeteksi.")

    st.markdown('</div>', unsafe_allow_html=True)
