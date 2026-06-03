"""
pages/preprocessing.py — Text Preprocessing page for MindScan.
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html

from utils import stemmer, clean_text


def render(emotion_df, stress_df, slang_dict):
    st.title("⚙️ Text Preprocessing")

    # ──────────────────────────────────────────
    # PREPROCESSING STEPS OVERVIEW
    # ──────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Tahapan Preprocessing")

    # --- Tokenization ---
    st.markdown("##### 🔠 Tokenization")
    sample_text = st.text_input(
        "Masukkan teks untuk tokenization:",
        value="Aku capek banget sama tugas kuliah hari ini",
        key="token_vis",
    )
    if sample_text:
        tokens = sample_text.lower().split()
        token_html = "".join(
            f"""<span style="background:#1e1e40;border:1px solid #6366f1;color:#c8c8ff;
                padding:8px 12px;border-radius:10px;margin:5px;display:inline-block;
                font-family:DM Mono;font-size:12px;font-weight:700;">{t}</span>"""
            for t in tokens
        )
        full_html = f"""
        <div style="background:#0d0d20;border:1px solid #2a2a5a;border-radius:16px;padding:18px;margin-top:10px;">
            {token_html}
        </div>
        """
        html(full_html, height=90)

    # --- Stemming ---
    st.markdown("##### 🌱 Stemming")
    stemming_text = st.text_input(
        "Masukkan teks untuk stemming:",
        value="Saya sedang memikirkan pekerjaan dan berlarian mencari solusi",
        key="stem_vis",
    )
    if stemming_text:
        original_tokens = stemming_text.lower().split()
        stemmed_tokens = stemmer.stem(stemming_text).split()
        max_len = min(len(original_tokens), len(stemmed_tokens))

        rows_html = "".join(
            f"""<tr>
                <td style="padding:12px;border-bottom:1px solid #1e1e40;color:#ccccff;">{original_tokens[i]}</td>
                <td style="padding:12px;border-bottom:1px solid #1e1e40;color:#22c55e;font-weight:700;">{stemmed_tokens[i]}</td>
                <td style="padding:12px;border-bottom:1px solid #1e1e40;color:#9999aa;">
                    {"✓ Changed" if original_tokens[i] != stemmed_tokens[i] else "-"}
                </td>
            </tr>"""
            for i in range(max_len)
        )
        table_html = f"""
        <div style="background:#0d0d20;border:1px solid #2a2a5a;border-radius:16px;overflow:hidden;margin-top:12px;">
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="background:#1e1e40;color:#ffffff;">
                        <th style="padding:14px;">Original</th>
                        <th style="padding:14px;">Stemmed</th>
                        <th style="padding:14px;">Status</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        html(table_html, height=350)

    st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # STEP CARDS
    # ──────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("①", "Lowercase",        "Huruf kecil semua"),
        ("②", "Remove URLs",      "http:// & www"),
        ("③", "Remove @/#",       "@mention #hashtag"),
        ("④", "Remove Non-Alpha", "Angka & simbol"),
        ("⑤", "Strip Spaces",     "Whitespace ekstra"),
    ]
    for col, (icon, step, desc) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"""
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:14px;
                        padding:16px 12px;text-align:center;'>
                <div style='font-size:20px;color:#a78bfa;'>{icon}</div>
                <div style='font-size:13px;font-weight:700;color:#e0e0ff;margin:6px 0;'>{step}</div>
                <div style='font-size:11px;color:#6666aa;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # SAMPLE COMPARISON + MANUAL TEST
    # ──────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Sample Perbandingan Teks (Emosi)")
        sample = emotion_df.sample(min(8, len(emotion_df)), random_state=1)
        for _, row in sample.iterrows():
            st.markdown(f"""
            <div style='margin:10px 0;padding:12px 16px;background:#0d0d20;
                        border-radius:10px;border:1px solid #1e1e3f;'>
                <div style='font-size:11px;color:#f59e0b;font-family:DM Mono;margin-bottom:6px;'>ORIGINAL</div>
                <div style='color:#aaaacc;font-size:13px;margin-bottom:8px;'>{str(row['text'])[:120]}</div>
                <div style='font-size:11px;color:#22c55e;font-family:DM Mono;margin-bottom:6px;'>CLEANED</div>
                <div style='color:#e0e0ff;font-size:13px;'>{str(row['clean_text'])[:120]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### Coba Preprocessing Manual")
        user_text = st.text_area(
            "Masukkan teks untuk dibersihkan:", height=100,
            placeholder="Ketik teks apapun di sini...",
        )
        if user_text:
            cleaned = clean_text(user_text, slang_dict)
            removed = set(user_text.lower().split()) - set(cleaned.split())
            st.markdown(f"""
            <div style='background:#0a200a;border:1px solid rgba(34,197,94,0.3);
                        border-radius:12px;padding:16px;margin-top:10px;'>
                <div style='font-size:11px;color:#22c55e;font-family:DM Mono;margin-bottom:8px;'>HASIL CLEANING</div>
                <div style='color:#e0e0ff;font-size:14px;line-height:1.6;'>
                    {cleaned if cleaned else '(teks kosong setelah cleaning)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if removed:
                badges = "".join(
                    f'<span style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);'
                    f'color:#f87171;padding:2px 8px;border-radius:6px;font-size:11px;margin:2px;'
                    f'font-family:DM Mono;display:inline-block;">{w}</span>'
                    for w in list(removed)[:15]
                )
                st.markdown(
                    f"<div style='margin-top:10px;'>"
                    f"<span style='font-size:11px;color:#7878aa;'>Token dihapus:</span> {badges}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Statistik Teks")
        avg_len_emotion = emotion_df["text"].apply(lambda x: len(str(x).split())).mean()
        avg_len_stress  = stress_df["text"].apply(lambda x: len(str(x).split())).mean()
        total_vocab     = len(set(" ".join(emotion_df["clean_text"]).split()))
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px;'>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:12px;padding:16px;text-align:center;'>
                <div style='font-size:11px;color:#7878aa;font-family:DM Mono;'>AVG LEN EMOSI</div>
                <div style='font-size:22px;font-weight:800;color:#a78bfa;'>{avg_len_emotion:.0f}</div>
                <div style='font-size:10px;color:#5a5a8a;'>kata/tweet</div>
            </div>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:12px;padding:16px;text-align:center;'>
                <div style='font-size:11px;color:#7878aa;font-family:DM Mono;'>AVG LEN STRES</div>
                <div style='font-size:22px;font-weight:800;color:#6366f1;'>{avg_len_stress:.0f}</div>
                <div style='font-size:10px;color:#5a5a8a;'>kata/post</div>
            </div>
            <div style='background:#13132a;border:1px solid #2a2a5a;border-radius:12px;padding:16px;text-align:center;'>
                <div style='font-size:11px;color:#7878aa;font-family:DM Mono;'>VOCAB SIZE</div>
                <div style='font-size:22px;font-weight:800;color:#22c55e;'>{total_vocab:,}</div>
                <div style='font-size:10px;color:#5a5a8a;'>unik token</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
