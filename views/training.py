"""
views/training.py — Model Training page for MindScan.
Anti-overfit pipeline:
  1. stratified split  2. TF-IDF fit on train only
  3. balance train only  4. train  5. eval + 5-fold CV
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit.components.v1 import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from models import build_model, apply_balancing, STRATEGY_INFO
from data_loader import STRESS_LABEL_MAP


def render(emotion_df, stress_df):
    st.title("🤖 Model Training")
    _model_comparison()

    col_cfg, col_info = st.columns([1, 1.3])
    with col_cfg:
        model_name, balance_strategy, tfidf_features, test_size, train_btn = _config()
    with col_info:
        _balance_preview(stress_df, balance_strategy)

    if train_btn:
        _run(emotion_df, stress_df, model_name, balance_strategy, tfidf_features, test_size)
    elif st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.markdown(
            f"<div style='background:#111128;border:1px dashed #2a2a5a;border-radius:14px;"
            f"padding:16px 22px;margin-top:16px;'><span style='color:#6666aa;font-size:13px;'>"
            f"📌 Model terakhir: <b style='color:#a78bfa'>{m['model_name']}</b> · "
            f"<b style='color:#a78bfa'>{m['balance']}</b> — "
            f"Accuracy <b style='color:#22c55e'>{m['acc_s']:.1%}</b> · "
            f"F1 <b style='color:#22c55e'>{m['f1_s']:.1%}</b></span></div>",
            unsafe_allow_html=True,
        )


# ── MODEL COMPARISON ─────────────────────────────────────────────────────────
def _model_comparison():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📋 Model Comparison")
    cards = [
        ("#6366f1","Logistic Regression",
         "Cepat & mudah diinterpretasi. C=0.1 (regularisasi kuat) mencegah model hafal vocab training.",
         "Kurang optimal untuk pola non-linear kompleks."),
        ("#22c55e","Naive Bayes",
         "Sangat cepat. Alpha=1.0 (Laplace smoothing) robust terhadap kata yang belum pernah dilihat.",
         "Asumsi independensi fitur tidak selalu realistis."),
        ("#f59e0b","Linear SVM",
         "Terbaik untuk teks berdimensi tinggi. C=0.1 memperlebar margin agar tahan noise vocab.",
         "Training lebih lambat dari LR dan NB."),
    ]
    body = '<div style="display:flex;flex-direction:column;gap:14px;margin-top:12px;">'
    for color, name, strength, weakness in cards:
        body += f"""
        <div style="background:#13132a;border:1px solid #2a2a5a;
                    border-left:4px solid {color};border-radius:14px;padding:16px;">
            <div style="color:#e0e0ff;font-size:16px;font-weight:800;margin-bottom:10px;">{name}</div>
            <div style="color:#6666aa;font-size:10px;font-family:DM Mono;margin-bottom:3px;">STRENGTH</div>
            <div style="color:#c8c8ff;font-size:12px;line-height:1.5;margin-bottom:8px;">{strength}</div>
            <div style="color:#6666aa;font-size:10px;font-family:DM Mono;margin-bottom:3px;">WEAKNESS</div>
            <div style="color:#aaaacc;font-size:12px;line-height:1.5;">{weakness}</div>
        </div>"""
    body += "</div>"
    html(body, height=520)
    st.markdown('</div>', unsafe_allow_html=True)


# ── CONFIG PANEL ─────────────────────────────────────────────────────────────
def _config():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### ⚙️ Konfigurasi Training")

    model_name = st.selectbox("Algoritma Model",
        ["Logistic Regression", "Naive Bayes", "Linear SVM"])
    balance_strategy = st.selectbox("Strategi Balancing Data",
        ["Random Oversampling","SMOTE","Random Undersampling","SMOTETomek","Tanpa Balancing"],
        help="Diterapkan hanya pada training set setelah split")
    tfidf_features = st.slider("TF-IDF Max Features", 1000, 20000, 5000, 500,
        help="Rekomendasi: 3000–7000. Lebih banyak ≠ lebih akurat.")
    test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)

    info_text, info_color = STRATEGY_INFO[balance_strategy]
    st.markdown(
        f"<div style='background:rgba(99,102,241,0.07);border-left:3px solid {info_color};"
        f"border-radius:0 8px 8px 0;padding:10px 14px;margin-top:8px;'>"
        f"<span style='color:#9999cc;font-size:12px;'>{info_text}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);
                border-radius:10px;padding:10px 14px;margin-top:10px;'>
        <div style='font-size:10px;color:#22c55e;font-family:DM Mono;margin-bottom:5px;'>💡 TIPS</div>
        <div style='font-size:11px;color:#7878aa;line-height:1.7;'>
            • <b style='color:#c8c8ff;'>Logistic Regression</b> = generalisasi terbaik<br>
            • TF-IDF <b style='color:#c8c8ff;'>3000–7000</b> lebih robust dari 10k+<br>
            • <b style='color:#c8c8ff;'>SMOTETomek</b> = boundary kelas paling bersih<br>
            • Accuracy >95% = waspadai overfit, cek CV Score
        </div>
    </div>
    """, unsafe_allow_html=True)

    train_btn = st.button("🚀 Train Model Sekarang", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return model_name, balance_strategy, tfidf_features, test_size, train_btn


# ── BALANCE PREVIEW ──────────────────────────────────────────────────────────
def _balance_preview(stress_df, balance_strategy):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📊 Distribusi Sebelum & Sesudah Balancing")

    vc_orig    = stress_df["stress_label"].value_counts().sort_index()
    bar_colors = ["#22c55e", "#f59e0b", "#ef4444"]

    if balance_strategy != "Tanpa Balancing":
        try:
            from models import apply_balancing
            tv = TfidfVectorizer(max_features=2000, sublinear_tf=True, min_df=2)
            Xp = tv.fit_transform(stress_df["clean_text"])
            _, yb = apply_balancing(Xp, stress_df["stress_label"], balance_strategy)
            vc_bal = pd.Series(yb).value_counts().sort_index()
        except Exception:
            vc_bal = vc_orig
    else:
        vc_bal = vc_orig

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, vc, title in [(axes[0], vc_orig, "Sebelum"), (axes[1], vc_bal, f"Setelah {balance_strategy}")]:
        xl = [STRESS_LABEL_MAP.get(int(k), str(k)) for k in vc.index]
        bars = ax.bar(xl, vc.values, color=bar_colors[:len(vc)], width=0.5, edgecolor="none")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(title, fontsize=10, pad=8)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{int(v):,}",
                    ha="center", fontsize=8, fontweight="bold", color="white")
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()

    if len(vc_bal) > 1:
        mx, mn = vc_bal.max(), vc_bal.min()
        ratio  = mx/mn if mn > 0 else float("inf")
        bpct   = (1-(mx-mn)/mx)*100
        st.markdown(
            f"<div style='display:flex;gap:8px;margin-top:10px;'>"
            f"<div class='balance-pill'>Ratio: {ratio:.2f}x</div>"
            f"<div class='balance-pill'>Balance: {bpct:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ── TRAINING ─────────────────────────────────────────────────────────────────
def _run(emotion_df, stress_df, model_name, balance_strategy, tfidf_features, test_size):
    st.markdown("<br>", unsafe_allow_html=True)
    ph = st.empty()

    steps = [
        "Memuat data",
        "Split train/test (stratified)",
        "Fit TF-IDF pada training set",
        f"Balancing training: {balance_strategy}",
        f"Training: {model_name}",
        "Evaluasi + Cross-Validation",
        "Selesai ✓",
    ]

    def prog(done, cur=None):
        b = ('<div class="section-card"><div style="font-size:13px;font-weight:700;'
             'color:#e0e0ff;margin-bottom:12px;">⏳ Training Progress</div>')
        for s in steps:
            if s in done:    icon, cls = "✅", "step-done"
            elif s == cur:   icon, cls = "⟳", "step-active"
            else:            icon, cls = "○", ""
            b += f'<div class="step-indicator {cls}">{icon} {s}</div>'
        return b + "</div>"

    done = []

    # 1. load
    ph.markdown(prog(done, steps[0]), unsafe_allow_html=True); time.sleep(0.2)
    done.append(steps[0])

    # 2. split — stratified
    ph.markdown(prog(done, steps[1]), unsafe_allow_html=True); time.sleep(0.2)
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        stress_df["clean_text"], stress_df["stress_label"],
        test_size=test_size, random_state=42, stratify=stress_df["stress_label"],
    )
    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(
        emotion_df["clean_text"], emotion_df["label"],
        test_size=test_size, random_state=42, stratify=emotion_df["label"],
    )
    done.append(steps[1])

    # 3. TF-IDF fit on train only
    ph.markdown(prog(done, steps[2]), unsafe_allow_html=True); time.sleep(0.2)
    tfidf = TfidfVectorizer(
        max_features=tfidf_features, ngram_range=(1,2),
        sublinear_tf=True, min_df=2, max_df=0.95,
    )
    X_tr_sv = tfidf.fit_transform(X_tr_s)
    X_te_sv = tfidf.transform(X_te_s)
    done.append(steps[2])

    # 4. balance train only
    ph.markdown(prog(done, steps[3]), unsafe_allow_html=True); time.sleep(0.2)
    try:
        X_bal, y_bal = apply_balancing(X_tr_sv, y_tr_s, balance_strategy)
    except Exception as ex:
        st.warning(f"Balancing gagal: {ex}. Menggunakan data original.")
        X_bal, y_bal = X_tr_sv, y_tr_s
    done.append(steps[3])

    # 5. train
    ph.markdown(prog(done, steps[4]), unsafe_allow_html=True); time.sleep(0.2)
    stress_model = build_model(model_name)
    stress_model.fit(X_bal, y_bal)

    emotion_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_features, ngram_range=(1,2),
            sublinear_tf=True, min_df=2, max_df=0.95,
        )),
        ("clf", build_model(model_name)),
    ])
    emotion_pipeline.fit(X_tr_e, y_tr_e)
    done.append(steps[4])

    # 6. evaluate
    ph.markdown(prog(done, steps[5]), unsafe_allow_html=True)
    pred_s = stress_model.predict(X_te_sv)
    pred_e = emotion_pipeline.predict(X_te_e)

    acc_s  = accuracy_score(y_te_s,  pred_s)
    prec_s = precision_score(y_te_s, pred_s, average="weighted", zero_division=0)
    rec_s  = recall_score(y_te_s,    pred_s, average="weighted", zero_division=0)
    f1_s   = f1_score(y_te_s,        pred_s, average="weighted", zero_division=0)
    acc_e  = accuracy_score(y_te_e,  pred_e)

    cv = cross_val_score(build_model(model_name), X_tr_sv, y_tr_s, cv=5, scoring="f1_weighted")
    train_acc = accuracy_score(y_tr_s, stress_model.predict(X_bal))

    done += [steps[5], steps[6]]
    ph.markdown(prog(done), unsafe_allow_html=True)

    st.session_state.emotion_model = emotion_pipeline
    st.session_state.stress_model  = (stress_model, tfidf)
    st.session_state.last_metrics  = dict(
        acc_s=acc_s, prec_s=prec_s, rec_s=rec_s, f1_s=f1_s,
        acc_e=acc_e, model_name=model_name, balance=balance_strategy,
    )

    # overfit banner
    gap = train_acc - acc_s
    if gap > 0.15:
        st.markdown(
            f"<div style='background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.3);"
            f"border-radius:12px;padding:12px 16px;margin-top:14px;'>"
            f"<b style='color:#ef4444;font-size:12px;'>⚠️ Potensi Overfit</b> — "
            f"<span style='color:#aaaacc;font-size:12px;'>train {train_acc:.1%} vs test {acc_s:.1%} "
            f"(gap {gap:.1%}). Coba kurangi TF-IDF features atau pakai SMOTETomek.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);"
            f"border-radius:12px;padding:12px 16px;margin-top:14px;'>"
            f"<b style='color:#22c55e;font-size:12px;'>✓ Generalisasi Baik</b> — "
            f"<span style='color:#aaaacc;font-size:12px;'>train {train_acc:.1%} vs test {acc_s:.1%} "
            f"(gap {gap:.1%})</span></div>",
            unsafe_allow_html=True,
        )

    # ── Stress metrics ──
    st.markdown("## 📊 Stress Model Evaluation")
    _metric_cards(acc_s, prec_s, rec_s, f1_s, "stress model")

    # CV card
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#13132a,#1a1a35);border:1px solid #2a2a5a;"
        f"border-radius:14px;padding:16px 20px;margin:14px 0;'>"
        f"<div style='font-size:10px;color:#6666aa;font-family:DM Mono;margin-bottom:6px;'>"
        f"5-FOLD CV F1 (training set)</div>"
        f"<span style='font-size:1.9rem;font-weight:800;color:#a78bfa;'>{cv.mean():.1%}</span>"
        f"<span style='font-size:12px;color:#6666aa;margin-left:10px;'>± {cv.std():.1%}</span>"
        f"<div style='margin-top:8px;display:flex;gap:5px;flex-wrap:wrap;'>"
        + "".join(
            f"<span style='background:#1e1e40;border:1px solid #3a3a6a;color:#c8c8ff;"
            f"padding:2px 9px;border-radius:6px;font-size:10px;font-family:DM Mono;'>"
            f"fold {i+1}: {s:.1%}</span>"
            for i, s in enumerate(cv)
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )

    _tfidf_chart(tfidf, X_tr_sv)
    if model_name in ["Logistic Regression", "Linear SVM"]:
        _feature_importance(stress_model, tfidf)

    c1, c2 = st.columns(2)
    with c1: _confusion_matrix(y_te_s, pred_s, stress_df["stress_label"].unique(), STRESS_LABEL_MAP)
    with c2: _class_report(y_te_s, pred_s)

    # ── Emotion metrics ──
    st.markdown('<hr style="border:1px solid rgba(99,102,241,0.2);margin:28px 0 22px;">', unsafe_allow_html=True)
    st.markdown("## 📊 Emotion Model Evaluation")
    acc_e2 = accuracy_score(y_te_e, pred_e)
    prec_e = precision_score(y_te_e, pred_e, average="weighted", zero_division=0)
    rec_e  = recall_score(y_te_e,   pred_e, average="weighted", zero_division=0)
    f1_e   = f1_score(y_te_e,       pred_e, average="weighted", zero_division=0)
    _metric_cards(acc_e2, prec_e, rec_e, f1_e, "emotion model")
    e_ticks = sorted(emotion_df["label"].unique())
    ec1, ec2 = st.columns(2)
    with ec1: _confusion_matrix(y_te_e, pred_e, e_ticks, {v:v for v in e_ticks})
    with ec2: _class_report(y_te_e, pred_e)


# ── HELPERS ──────────────────────────────────────────────────────────────────
def _metric_cards(acc, prec, rec, f1, sub):
    st.markdown(f"""
    <div class="bento-grid" style="margin-top:18px;">
        <div class="bento-card"><div class="bento-label">Accuracy</div>
            <div class="bento-value">{acc:.1%}</div><div class="bento-sub">{sub}</div></div>
        <div class="bento-card"><div class="bento-label">Precision</div>
            <div class="bento-value">{prec:.1%}</div><div class="bento-sub">weighted</div></div>
        <div class="bento-card"><div class="bento-label">Recall</div>
            <div class="bento-value">{rec:.1%}</div><div class="bento-sub">weighted</div></div>
        <div class="bento-card"><div class="bento-label">F1-Score</div>
            <div class="bento-value">{f1:.1%}</div><div class="bento-sub">weighted</div></div>
    </div>
    """, unsafe_allow_html=True)


def _tfidf_chart(tfidf, X_vec):
    st.markdown("##### 🔍 TF-IDF Top Keywords")
    scores = np.asarray(X_vec.mean(axis=0)).ravel()
    idx    = scores.argsort()[-15:][::-1]
    words  = [tfidf.get_feature_names_out()[i] for i in idx]
    vals   = [scores[i] for i in idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(words[::-1], vals[::-1])
    ax.set_xlabel("Avg TF-IDF Score")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()


def _feature_importance(model, tfidf):
    st.markdown("##### 🧠 Top Influential Words per Kelas")
    fn = tfidf.get_feature_names_out()
    try:
        coef = model.coef_
        labels = {0:("😌 Normal","#22c55e"), 1:("😥 Mild","#f59e0b"), 2:("😫 High","#ef4444")}
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            with col:
                name, _ = labels[idx]
                top = np.argsort(coef[idx])[-10:]
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.barh([fn[i] for i in top], [coef[idx][i] for i in top])
                ax.set_title(name, fontsize=11)
                ax.spines[["top","right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig); plt.close()
    except Exception as e:
        st.warning(f"Feature importance tidak tersedia: {e}")


def _confusion_matrix(y_true, y_pred, unique_labels, label_map):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 🎯 Confusion Matrix")
    cm    = confusion_matrix(y_true, y_pred)
    ticks = [label_map.get(k, str(k)) for k in sorted(unique_labels)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", cbar=True,
                xticklabels=ticks[:cm.shape[1]], yticklabels=ticks[:cm.shape[0]],
                linewidths=2, linecolor="#0f0f1f", square=True,
                annot_kws={"size":14,"weight":"bold","color":"white"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11, color="#c8c8ff", labelpad=10)
    ax.set_ylabel("Actual",    fontsize=11, color="#c8c8ff", labelpad=10)
    ax.tick_params(colors="#ffffff", labelsize=9)
    ax.set_facecolor("#111128"); fig.patch.set_facecolor("#111128")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


def _class_report(y_true, y_pred):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📋 Classification Report")
    rdf = (pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).transpose().round(3).reset_index())
    rdf.columns = ["Class","Precision","Recall","F1-Score","Support"]
    h1,h2,h3,h4,h5 = st.columns([2,1,1,1,1])
    for col, lbl in zip([h1,h2,h3,h4,h5], ["*Class*","*Prec*","*Rec*","*F1*","*Sup*"]):
        col.markdown(lbl)
    st.markdown("---")
    for _, row in rdf.iterrows():
        c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
        c1.markdown(f"<span style='color:#c8c8ff;font-weight:700'>{row['Class']}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#818cf8;font-family:monospace'>{row['Precision']:.3f}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:#22c55e;font-family:monospace'>{row['Recall']:.3f}</span>", unsafe_allow_html=True)
        c4.markdown(f"<span style='color:#f59e0b;font-family:monospace'>{row['F1-Score']:.3f}</span>", unsafe_allow_html=True)
        c5.markdown(f"<span style='color:#ef4444;font-family:monospace'>{int(row['Support'])}</span>", unsafe_allow_html=True)
        st.markdown("<div class='report-row'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
