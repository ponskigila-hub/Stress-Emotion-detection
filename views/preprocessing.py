"""
views/preprocessing.py — Preprocessing page for MindScan.
"""

import streamlit as st
from streamlit.components.v1 import html

from utils import stemmer, clean_text


def render(emotion_df, stress_df, slang_dict):
    st.title("⚙️ Text Preprocessing")

    # ── Pipeline steps overview ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Tahapan Preprocessing")

    # Tokenization
    st.markdown("##### 🔠 Tokenization")
    sample = st.text_input("Masukkan teks untuk tokenization:",
        value="Aku capek banget sama tugas kuliah hari ini", key="tok_vis")
    if sample:
        tokens   = sample.lower().split()
        badges   = "".join(
            f"<span style='background:#1e1e40;border:1px solid #6366f1;color:#c8c8ff;"
            f"padding:7px 11px;border-radius:9px;margin:4px;display:inline-block;"
            f"font-family:DM Mono;font-size:12px;font-weight:700;'>{t}</span>"
            for t in tokens
        )
        html(f"<div style='background:#0d0d20;border:1px solid #2a2a5a;"
             f"border-radius:14px;padding:16px;margin-top:8px;'>{badges}</div>",
             height=80)

    # Stemming
    st.markdown("##### 🌱 Stemming")
    stem_text = st.text_input("Masukkan teks untuk stemming:",
        value="Saya sedang memikirkan pekerjaan dan berlarian mencari solusi", key="stem_vis")
    if stem_text:
        orig    = stem_text.lower().split()
        stemmed = stemmer.stem(stem_text).split()
        rows    = "".join(
            f"<tr><td style='padding:10px;border-bottom:1px solid #1e1e40;color:#ccccff;'>{orig[i]}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #1e1e40;color:#22c55e;font-weight:700;'>{stemmed[i]}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #1e1e40;color:#9999aa;'>"
            f"{'✓ Changed' if orig[i]!=stemmed[i] else '—'}</td></tr>"
            for i in range(min(len(orig), len(stemmed)))
        )
        html(f"""
        <div style='background:#0d0d20;border:1px solid #2a2a5a;border-radius:14px;
                    overflow:hidden;margin-top:10px;'>
            <table style='width:100%;border-collapse:collapse;'>
                <thead>
                    <tr style='background:#1e1e40;color:#fff;'>
                        <th style='padding:12px;'>Original</th>
                        <th style='padding:12px;'>Stemmed</th>
                        <th style='padding:12px;'>Status</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>""", height=320)

    st.markdown('</div>', unsafe_allow_html=True)

    # Step cards
    steps = [
        ("①","Lowercase","Huruf kecil semua"),
        ("②","Remove URLs","http:// & www"),
        ("③","Remove @/#","mention & hashtag"),
        ("④","Remove Non-Alpha","Angka & simbol"),
        ("⑤","Normalize Slang","Kata gaul → formal"),
        ("⑥","Remove Stopwords","Kata tidak penting (dengan KEEP_WORDS)"),
        ("⑦","Stemming","Sastrawi stemmer"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:12px;
                        padding:14px 10px;text-align:center;'>
                <div style='font-size:18px;color:#a78bfa;'>{icon}</div>
                <div style='font-size:12px;font-weight:700;color:#e0e0ff;margin:5px 0;'>{title}</div>
                <div style='font-size:10px;color:#6666aa;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Before / After + Manual test
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Sample Perbandingan Teks (Emosi)")
        for _, row in emotion_df.sample(min(6, len(emotion_df)), random_state=1).iterrows():
            st.markdown(f"""
            <div style='margin:8px 0;padding:12px;background:#0d0d20;
                        border-radius:10px;border:1px solid #1e1e3f;'>
                <div style='font-size:10px;color:#f59e0b;font-family:DM Mono;margin-bottom:4px;'>ORIGINAL</div>
                <div style='color:#aaaacc;font-size:12px;margin-bottom:7px;'>{str(row["text"])[:120]}</div>
                <div style='font-size:10px;color:#22c55e;font-family:DM Mono;margin-bottom:4px;'>CLEANED</div>
                <div style='color:#e0e0ff;font-size:12px;'>{str(row["clean_text"])[:120]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Coba Preprocessing Manual")
        user_text = st.text_area("Masukkan teks:", height=100,
            placeholder="Ketik teks apapun di sini...")
        if user_text:
            cleaned = clean_text(user_text, slang_dict)
            removed = set(user_text.lower().split()) - set(cleaned.split())
            st.markdown(f"""
            <div style='background:#0a200a;border:1px solid rgba(34,197,94,0.3);
                        border-radius:10px;padding:14px;margin-top:8px;'>
                <div style='font-size:10px;color:#22c55e;font-family:DM Mono;margin-bottom:6px;'>HASIL CLEANING</div>
                <div style='color:#e0e0ff;font-size:13px;line-height:1.6;'>
                    {cleaned if cleaned else "(teks kosong setelah cleaning)"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if removed:
                badges = "".join(
                    f"<span style='background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);"
                    f"color:#f87171;padding:2px 7px;border-radius:5px;font-size:10px;margin:2px;"
                    f"font-family:DM Mono;display:inline-block;'>{w}</span>"
                    for w in list(removed)[:15]
                )
                st.markdown(
                    f"<div style='margin-top:8px;'><span style='font-size:10px;color:#7878aa;'>"
                    f"Token dihapus:</span> {badges}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Statistik Teks")
        avg_e = emotion_df["text"].apply(lambda x: len(str(x).split())).mean()
        avg_s = stress_df["text"].apply(lambda x: len(str(x).split())).mean()
        vocab = len(set(" ".join(emotion_df["clean_text"]).split()))
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px;'>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:10px;
                        padding:14px;text-align:center;'>
                <div style='font-size:10px;color:#7878aa;font-family:DM Mono;'>AVG EMOSI</div>
                <div style='font-size:20px;font-weight:800;color:#a78bfa;'>{avg_e:.0f}</div>
                <div style='font-size:9px;color:#5a5a8a;'>kata/tweet</div>
            </div>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:10px;
                        padding:14px;text-align:center;'>
                <div style='font-size:10px;color:#7878aa;font-family:DM Mono;'>AVG STRES</div>
                <div style='font-size:20px;font-weight:800;color:#6366f1;'>{avg_s:.0f}</div>
                <div style='font-size:9px;color:#5a5a8a;'>kata/post</div>
            </div>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:10px;
                        padding:14px;text-align:center;'>
                <div style='font-size:10px;color:#7878aa;font-family:DM Mono;'>VOCAB</div>
                <div style='font-size:20px;font-weight:800;color:#22c55e;'>{vocab:,}</div>
                <div style='font-size:9px;color:#5a5a8a;'>unik token</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
