"""
views/eda.py — EDA page for MindScan.
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
    t1, t2, t3, t4 = st.tabs([
        "   Emosi   ", "   Stres   ", "   WordCloud   ", "   Slang Analysis   "
    ])
    with t1: _emotion_tab(emotion_df)
    with t2: _stress_tab(stress_df)
    with t3: _wordcloud_tab(emotion_df, stress_df)
    with t4: _slang_tab(emotion_df, stress_df, slang_words)


def _emotion_tab(emotion_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Preview Data Emosi")
    st.dataframe(emotion_df[["text","label","clean_text"]].head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Distribusi Kelas Emosi")
    vc      = emotion_df["label"].value_counts()
    palette = sns.color_palette("husl", len(vc))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    bars = axes[0].barh(vc.index, vc.values, color=palette, edgecolor="none", height=0.6)
    axes[0].spines[["top","right","left"]].set_visible(False)
    axes[0].set_xlabel("Jumlah")
    for bar, val in zip(bars, vc.values):
        axes[0].text(val+3, bar.get_y()+bar.get_height()/2,
                     f"{val:,}", va="center", fontsize=9, color="white")

    wedges, texts, autos = axes[1].pie(
        vc.values, labels=vc.index, autopct="%1.1f%%", colors=palette,
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="#09090f", linewidth=2),
    )
    for at in autos: at.set_fontsize(9); at.set_color("white")
    axes[1].set_facecolor("#111128")
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


def _stress_tab(stress_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Preview Data Stres")
    st.dataframe(stress_df[["text","stress_label","clean_text"]].head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    bar_colors = ["#22c55e","#f59e0b","#ef4444"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Distribusi Label Stres")
        vc = stress_df["stress_label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        xl   = [STRESS_LABEL_MAP.get(int(k), str(k)) for k in vc.index]
        bars = ax.bar(xl, vc.values, color=bar_colors[:len(vc)], width=0.5, edgecolor="none")
        ax.spines[["top","right"]].set_visible(False)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:,}",
                    ha="center", fontsize=10, fontweight="bold", color="white")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Panjang Teks (Distribusi)")
        stress_df = stress_df.copy()
        stress_df["text_len"] = stress_df["text"].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots(figsize=(5, 4))
        for lbl, color in [(0,"#22c55e"),(1,"#f59e0b"),(2,"#ef4444")]:
            sub = stress_df[stress_df["stress_label"]==lbl]["text_len"]
            if len(sub):
                ax.hist(sub, bins=30, alpha=0.6, color=color,
                        label=STRESS_LABEL_MAP.get(lbl,str(lbl)), edgecolor="none")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("Jumlah Kata"); ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def _wordcloud_tab(emotion_df, stress_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    choice = st.selectbox("Pilih Dataset untuk WordCloud",
        ["Semua Emosi","Normal (label 0)","Mild Stress (label 1)","High Stress (label 2)"])
    text_map = {
        "Semua Emosi":           " ".join(emotion_df["text"].astype(str)),
        "Normal (label 0)":      " ".join(stress_df[stress_df["stress_label"]==0]["text"].astype(str)),
        "Mild Stress (label 1)": " ".join(stress_df[stress_df["stress_label"]==1]["text"].astype(str)),
        "High Stress (label 2)": " ".join(stress_df[stress_df["stress_label"]==2]["text"].astype(str)),
    }
    cmap_map = {
        "Semua Emosi":"viridis","Normal (label 0)":"Greens",
        "Mild Stress (label 1)":"YlOrBr","High Stress (label 2)":"Reds",
    }
    text = text_map[choice]
    if text.strip():
        wc = WordCloud(width=1200, height=420, background_color="#111128",
                       colormap=cmap_map[choice], max_words=120,
                       prefer_horizontal=0.85).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig.patch.set_facecolor("#111128"); fig.tight_layout(pad=0)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("##### ☁️ Slang WordCloud")
    ds_choice = st.selectbox("Pilih Dataset Slang",
        ["Emotion Dataset","Stress Dataset"], key="slang_wc")
    src = emotion_df if ds_choice == "Emotion Dataset" else stress_df
    all_sl = [w for words in src["slang_words"] for w in words]
    wc_text = " ".join(all_sl)
    if wc_text.strip():
        wc = WordCloud(width=1200, height=380, background_color="#111128",
                       colormap="plasma").generate(wc_text)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.warning("Tidak ada slang terdeteksi.")
    st.markdown('</div>', unsafe_allow_html=True)


def _slang_tab(emotion_df, stress_df, slang_words):
    import pandas as pd
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 🔤 Analisis Slang Language")
    ds = st.selectbox("Pilih Dataset", ["Emotion Dataset","Stress Dataset"])
    df = emotion_df if ds == "Emotion Dataset" else stress_df

    total_sl  = int(df["slang_count"].sum())
    n         = len(df)
    with_sl   = int((df["slang_count"]>0).sum())
    pct       = with_sl/n*100

    try:
        st.html(f"""
        <div class="bento-grid">
            <div class="bento-card"><div class="bento-label">TOTAL SLANG</div>
                <div class="bento-value">{total_sl:,}</div><div class="bento-sub">kata ditemukan</div></div>
            <div class="bento-card"><div class="bento-label">TEXT WITH SLANG</div>
                <div class="bento-value">{with_sl:,}</div><div class="bento-sub">teks mengandung slang</div></div>
            <div class="bento-card"><div class="bento-label">PERCENTAGE</div>
                <div class="bento-value">{pct:.1f}%</div><div class="bento-sub">dari total teks</div></div>
            <div class="bento-card"><div class="bento-label">SLANG VOCAB</div>
                <div class="bento-value">{len(slang_words):,}</div><div class="bento-sub">kamus slang</div></div>
        </div>""")
    except Exception:
        st.markdown(f"""
        <div class="bento-grid">
            <div class="bento-card"><div class="bento-label">TOTAL SLANG</div>
                <div class="bento-value">{total_sl:,}</div></div>
            <div class="bento-card"><div class="bento-label">TEXT WITH SLANG</div>
                <div class="bento-value">{with_sl:,}</div></div>
            <div class="bento-card"><div class="bento-label">PERCENTAGE</div>
                <div class="bento-value">{pct:.1f}%</div></div>
            <div class="bento-card"><div class="bento-label">SLANG VOCAB</div>
                <div class="bento-value">{len(slang_words):,}</div></div>
        </div>""", unsafe_allow_html=True)

    all_sl = [w for words in df["slang_words"] for w in words]
    counter = Counter(all_sl)
    if counter:
        top = counter.most_common(15)
        sdf = pd.DataFrame(top, columns=["Slang","Frequency"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📊 Top Slang Words")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(sdf["Slang"][::-1], sdf["Frequency"][::-1])
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlabel("Frequency")
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("##### 🥧 Slang Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(sdf["Frequency"][:8], labels=sdf["Slang"][:8], autopct="%1.1f%%")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("##### 🔄 Before vs After Normalization")
        samples = df[df["slang_count"]>0].sample(min(6, with_sl), random_state=42)
        for _, row in samples.iterrows():
            card = f"""
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:14px;
                        padding:14px;margin:10px 0;'>
                <div style='font-size:10px;color:#f59e0b;font-family:DM Mono;margin-bottom:5px;'>ORIGINAL</div>
                <div style='font-size:12px;color:#ccccff;margin-bottom:10px;line-height:1.5;'>{escape(str(row["text"]))[:240]}</div>
                <div style='font-size:10px;color:#22c55e;font-family:DM Mono;margin-bottom:5px;'>NORMALIZED</div>
                <div style='font-size:12px;color:#fff;margin-bottom:8px;line-height:1.5;'>{escape(str(row["clean_text"]))[:240]}</div>
                <div style='font-size:10px;color:#8888aa;'>Slang: <span style='color:#a78bfa'>{", ".join(row["slang_words"])}</span></div>
            </div>"""
            try: st.html(card)
            except Exception: st.markdown(card, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada slang terdeteksi.")
    st.markdown('</div>', unsafe_allow_html=True)
