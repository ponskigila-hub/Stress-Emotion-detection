"""
pages/training.py — Model Training page for MindScan.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit.components.v1 import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from models import build_model, apply_balancing, STRATEGY_INFO
from data_loader import STRESS_LABEL_MAP


# ──────────────────────────────────────────
# MAIN RENDER
# ──────────────────────────────────────────
def render(emotion_df, stress_df):
    st.title("🤖 Model Training")

    _render_model_comparison()

    col_cfg, col_info = st.columns([1, 1.3])

    with col_cfg:
        model_name, balance_strategy, tfidf_features, test_size, train_btn = _render_config()

    with col_info:
        _render_balance_preview(stress_df, balance_strategy)

    # ── TRAINING ──
    if train_btn:
        _run_training(emotion_df, stress_df, model_name, balance_strategy, tfidf_features, test_size)

    elif st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.markdown(f"""
        <div style='background:#111128;border:1px dashed #2a2a5a;border-radius:14px;
                    padding:18px 24px;margin-top:16px;'>
            <span style='color:#6666aa;font-size:13px;'>
                📌 Model terakhir: <b style='color:#a78bfa'>{m['model_name']}</b> dengan
                <b style='color:#a78bfa'>{m['balance']}</b> —
                Accuracy: <b style='color:#22c55e'>{m['acc_s']:.1%}</b> |
                F1: <b style='color:#22c55e'>{m['f1_s']:.1%}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# MODEL COMPARISON CARDS
# ──────────────────────────────────────────
def _render_model_comparison():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📋 Model Comparison")

    comparison_html = """
    <div style="display:flex;flex-direction:column;gap:16px;margin-top:14px;">

        <div style="background:#13132a;border:1px solid #2a2a5a;border-left:4px solid #6366f1;border-radius:16px;padding:18px;">
            <div style="color:#e0e0ff;font-size:17px;font-weight:800;margin-bottom:14px;">Logistic Regression</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">STRENGTH</div>
            <div style="color:#c8c8ff;font-size:13px;line-height:1.6;margin-bottom:12px;">Cepat, ringan, dan mudah diinterpretasi untuk text classification.</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">WEAKNESS</div>
            <div style="color:#aaaacc;font-size:13px;line-height:1.6;">Kurang optimal dalam menangani pola non-linear kompleks.</div>
        </div>

        <div style="background:#13132a;border:1px solid #2a2a5a;border-left:4px solid #22c55e;border-radius:16px;padding:18px;">
            <div style="color:#e0e0ff;font-size:17px;font-weight:800;margin-bottom:14px;">Naive Bayes</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">STRENGTH</div>
            <div style="color:#c8c8ff;font-size:13px;line-height:1.6;margin-bottom:12px;">Sangat cepat dan efisien untuk klasifikasi teks berbasis TF-IDF.</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">WEAKNESS</div>
            <div style="color:#aaaacc;font-size:13px;line-height:1.6;">Mengasumsikan semua fitur independen sehingga kadang kurang realistis.</div>
        </div>

        <div style="background:#13132a;border:1px solid #2a2a5a;border-left:4px solid #f59e0b;border-radius:16px;padding:18px;">
            <div style="color:#e0e0ff;font-size:17px;font-weight:800;margin-bottom:14px;">Linear SVM</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">STRENGTH</div>
            <div style="color:#c8c8ff;font-size:13px;line-height:1.6;margin-bottom:12px;">Sangat baik untuk data teks berdimensi tinggi seperti TF-IDF.</div>
            <div style="color:#6666aa;font-size:11px;font-family:DM Mono;margin-bottom:4px;">WEAKNESS</div>
            <div style="color:#aaaacc;font-size:13px;line-height:1.6;">Training lebih lambat dibanding Logistic Regression dan Naive Bayes.</div>
        </div>

    </div>
    """
    html(comparison_html, height=550)
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# CONFIG PANEL
# ──────────────────────────────────────────
def _render_config():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### ⚙️ Konfigurasi Training")

    model_name = st.selectbox(
        "Algoritma Model",
        ["Logistic Regression", "Naive Bayes", "Linear SVM"],
        help="Pilih algoritma klasifikasi untuk model stres",
    )
    balance_strategy = st.selectbox(
        "Strategi Balancing Data",
        ["Random Oversampling", "SMOTE", "Random Undersampling", "Tanpa Balancing"],
        help="Strategi untuk menyeimbangkan distribusi kelas label",
    )
    tfidf_features = st.slider("TF-IDF Max Features", 1000, 20000, 10000, 1000)
    test_size      = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)

    info_text, info_color = STRATEGY_INFO[balance_strategy]
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.07);border-left:3px solid {info_color};
                border-radius:0 8px 8px 0;padding:12px 14px;margin-top:8px;'>
        <span style='color:#9999cc;font-size:12px;'>{info_text}</span>
    </div>
    """, unsafe_allow_html=True)

    train_btn = st.button("🚀 Train Model Sekarang", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return model_name, balance_strategy, tfidf_features, test_size, train_btn


# ──────────────────────────────────────────
# BALANCE PREVIEW
# ──────────────────────────────────────────
def _render_balance_preview(stress_df, balance_strategy):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📊 Distribusi Sebelum & Sesudah Balancing")

    vc_orig    = stress_df["stress_label"].value_counts().sort_index()
    bar_colors = ["#22c55e", "#f59e0b", "#ef4444"]

    if balance_strategy != "Tanpa Balancing":
        try:
            tfidf_prev = TfidfVectorizer(max_features=2000)
            X_prev     = tfidf_prev.fit_transform(stress_df["clean_text"])
            X_bal, y_bal = apply_balancing(X_prev, stress_df["stress_label"], balance_strategy)
            vc_bal = pd.Series(y_bal).value_counts().sort_index()
        except Exception:
            vc_bal = vc_orig
    else:
        vc_bal = vc_orig

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, vc_data, title in [
        (axes[0], vc_orig, "Sebelum Balancing"),
        (axes[1], vc_bal,  f"Setelah {balance_strategy}"),
    ]:
        x_l  = [STRESS_LABEL_MAP.get(int(k), str(k)) for k in vc_data.index]
        bars = ax.bar(x_l, vc_data.values, color=bar_colors[:len(vc_data)], width=0.5, edgecolor='none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, fontsize=11, pad=10)
        for bar, val in zip(bars, vc_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                    f"{int(val):,}", ha='center', fontsize=9, fontweight='bold', color='white')
    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    if len(vc_bal) > 1:
        max_v, min_v  = vc_bal.max(), vc_bal.min()
        ratio         = max_v / min_v if min_v > 0 else float('inf')
        balance_pct   = (1 - (max_v - min_v) / max_v) * 100
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-top:12px;'>
            <div class='balance-pill'>✓ Imbalance Ratio: {ratio:.2f}x</div>
            <div class='balance-pill'>✓ Balance Score: {balance_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# TRAINING EXECUTION
# ──────────────────────────────────────────
def _run_training(emotion_df, stress_df, model_name, balance_strategy, tfidf_features, test_size):
    st.markdown("<br>", unsafe_allow_html=True)
    progress_ph = st.empty()

    all_steps = [
        "Memuat data & preprocessing",
        "Split data (train/test)",
        "Fit TF-IDF pada TRAIN",
        f"Balancing pada TRAIN: {balance_strategy}",
        f"Training stress model: {model_name}",
        "Evaluasi model",
        "Selesai ✓",
    ]

    def render_progress(done, current=None):
        body = '<div class="section-card"><div style="font-size:14px;font-weight:700;color:#e0e0ff;margin-bottom:14px;">⏳ Training Progress</div>'
        for s in all_steps:
            if s in done:        icon, cls = "✅", "step-done"
            elif s == current:   icon, cls = "⟳", "step-active"
            else:                icon, cls = "○", ""
            body += f'<div class="step-indicator {cls}">{icon} {s}</div>'
        body += '</div>'
        return body

    done = []

    # Step 1: Load data
    current = all_steps[0]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)
    time.sleep(0.3)
    X_emotion = emotion_df["clean_text"]
    y_emotion = emotion_df["label"]
    X_stress_text = stress_df["clean_text"]
    y_stress = stress_df["stress_label"]
    done.append(current)

    # Step 2: Split stress data into train/test (before any vectorization)
    current = all_steps[1]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)
    time.sleep(0.3)
    X_train_s_text, X_test_s_text, y_train_s, y_test_s = train_test_split(
        X_stress_text, y_stress, test_size=test_size, random_state=42
    )
    done.append(current)

    # Step 3: Fit TF-IDF on TRAIN texts only, then transform both train and test
    current = all_steps[2]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)
    time.sleep(0.3)
    tfidf = TfidfVectorizer(max_features=tfidf_features, ngram_range=(1, 2))
    X_train_s_vec = tfidf.fit_transform(X_train_s_text)   # fit on train only
    X_test_s_vec  = tfidf.transform(X_test_s_text)        # transform test
    done.append(current)

    # Step 4: Apply balancing ONLY on training vectors and training labels
    current = all_steps[3]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)
    time.sleep(0.3)
    try:
        X_resampled, y_resampled = apply_balancing(X_train_s_vec, y_train_s, balance_strategy)
    except Exception as e:
        st.error(f"Balancing gagal: {e}. Menggunakan data original.")
        X_resampled, y_resampled = X_train_s_vec, y_train_s
    done.append(current)

    # Step 5: Train stress model on balanced training data
    current = all_steps[4]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)
    time.sleep(0.3)
    stress_model = build_model(model_name)
    stress_model.fit(X_resampled, y_resampled)
    done.append(current)

    # ========== EMOTION MODEL (konservatif, fallback untuk SVM) ==========
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_emotion, y_emotion, test_size=test_size, random_state=42
    )

    # Batasi fitur sangat ketat
    emotion_max_features = min(1000, tfidf_features)   # maksimal 1000 fitur
    emotion_tfidf = TfidfVectorizer(
        max_features=emotion_max_features,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=2,
        max_df=0.8
    )

    emotion_clf = build_model(model_name)

    emotion_pipeline = Pipeline([
        ("tfidf", emotion_tfidf),
        ("clf",   emotion_clf)
    ])
    emotion_pipeline.fit(X_train_e, y_train_e)
    # ==========================================================================

    # Step 6: Evaluate
    current = all_steps[5]
    progress_ph.markdown(render_progress(done, current), unsafe_allow_html=True)

    pred_stress  = stress_model.predict(X_test_s_vec)
    pred_emotion = emotion_pipeline.predict(X_test_e)

    acc_s  = round(accuracy_score(y_test_s,  pred_stress),  4)
    prec_s = round(precision_score(y_test_s, pred_stress,  average="weighted", zero_division=0), 4)
    rec_s  = round(recall_score(y_test_s,    pred_stress,  average="weighted", zero_division=0), 4)
    f1_s   = round(f1_score(y_test_s,        pred_stress,  average="weighted", zero_division=0), 4)
    acc_e  = round(accuracy_score(y_test_e,  pred_emotion), 4)

    done += [all_steps[5], all_steps[6]]
    progress_ph.markdown(render_progress(done), unsafe_allow_html=True)

    # For visualisation only: create a full TF-IDF matrix (using same tfidf fitted on train)
    X_full_vec = tfidf.transform(X_stress_text)

    # 🔍 Overfitting & Cross-Validation Analysis (Stress Model)
    with st.expander("🔍 Overfitting & Cross-Validation Analysis (Stress Model)", expanded=False):
        _render_overfitting_analysis(stress_model, X_resampled, y_resampled, X_test_s_vec, y_test_s, X_resampled, y_resampled, model_name)

    # Save to session
    st.session_state.emotion_model = emotion_pipeline
    st.session_state.stress_model  = (stress_model, tfidf)
    st.session_state.last_metrics  = {
        "acc_s": acc_s, "prec_s": prec_s, "rec_s": rec_s, "f1_s": f1_s,
        "acc_e": acc_e, "pred_stress": pred_stress, "y_test_s": y_test_s,
        "model_name": model_name, "balance": balance_strategy,
    }

    # ── Stress evaluation ──
    st.markdown("## Stress Model Evaluation")
    _render_metric_cards(acc_s, prec_s, rec_s, f1_s, "stress model")
    
    st.markdown("<br>", unsafe_allow_html=True)

    _render_tfidf_analysis(tfidf, X_full_vec)

    if model_name in ["Logistic Regression", "Linear SVM"]:
        _render_feature_importance(stress_model, tfidf)

    c1, c2 = st.columns(2)
    with c1:
        _render_confusion_matrix(y_test_s, pred_stress, stress_df["stress_label"].unique(), STRESS_LABEL_MAP)
    with c2:
        _render_classification_report(y_test_s, pred_stress)

    # ── Emotion evaluation ──
    st.markdown("""<hr style="border:1px solid rgba(99,102,241,0.2);margin:30px 0 25px;">""", unsafe_allow_html=True)
    st.markdown("## Emotion Model Evaluation")

    pred_emotion2 = emotion_pipeline.predict(X_test_e)
    acc_e2  = accuracy_score(y_test_e,  pred_emotion2)
    prec_e  = precision_score(y_test_e, pred_emotion2, average="weighted", zero_division=0)
    rec_e   = recall_score(y_test_e,    pred_emotion2, average="weighted", zero_division=0)
    f1_e    = f1_score(y_test_e,        pred_emotion2, average="weighted", zero_division=0)

    _render_metric_cards(acc_e2, prec_e, rec_e, f1_e, "emotion model")

    # 🔍 Overfitting & Cross-Validation for Emotion Model
    with st.expander("🔍 Overfitting & Cross-Validation Analysis (Emotion Model)", expanded=False):
        # Tampilkan nama model yang sebenarnya (fallback jika SVM)
        emotion_model_name = model_name if model_name != "Linear SVM" else "Logistic Regression (fallback)"
        _render_overfitting_analysis_emotion(
            emotion_pipeline,
            X_train_e, y_train_e,
            X_test_e, y_test_e,
            X_emotion, y_emotion,
            emotion_model_name
        )

    ec1, ec2 = st.columns(2)
    emotion_ticks = sorted(emotion_df["label"].unique())
    with ec1:
        _render_confusion_matrix(y_test_e, pred_emotion2, emotion_ticks, {v: v for v in emotion_ticks})
    with ec2:
        _render_classification_report(y_test_e, pred_emotion2)


# ──────────────────────────────────────────
# 🔍 OVERFITTING & CROSS-VALIDATION (Stress Model)
# ──────────────────────────────────────────
def _render_overfitting_analysis(model, X_train, y_train, X_test, y_test, X_full, y_full, model_name):
    """
    Computes train/test gap and performs k-fold cross-validation (5 folds)
    on the full (resampled) dataset. Displays metrics using bento cards
    and a styled recommendation box.
    """
    # Train accuracy (on the training split)
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test accuracy (already available)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    gap = train_acc - test_acc
    
    # Cross-validation on the full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
    cv_f1_scores  = cross_val_score(model, X_full, y_full, cv=cv, scoring='f1_weighted')
    
    # --- Metric Cards (same bento-grid as _render_metric_cards) ---
    st.markdown("""
    <div class="bento-grid" style="margin-top:0px; margin-bottom:20px;">
        <div class="bento-card">
            <div class="bento-label">📈 Train Accuracy</div>
            <div class="bento-value">{:.1%}</div>
            <div class="bento-sub">on training split</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">📉 Test Accuracy</div>
            <div class="bento-value">{:.1%}</div>
            <div class="bento-sub">gap = {:.1%}</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">🔁 CV Accuracy</div>
            <div class="bento-value">{:.1%} ± {:.1%}</div>
            <div class="bento-sub">5-fold stratified</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">🎯 CV F1 (weighted)</div>
            <div class="bento-value">{:.1%} ± {:.1%}</div>
            <div class="bento-sub">5-fold stratified</div>
        </div>
    </div>
    """.format(train_acc, test_acc, gap, cv_acc_scores.mean(), cv_acc_scores.std(), cv_f1_scores.mean(), cv_f1_scores.std()), unsafe_allow_html=True)
    
    # --- Plot with dark theme matching app ---
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#111128')
    ax.set_facecolor('#111128')
    ax.plot(range(1, 6), cv_acc_scores, 'o-', color='#f59e0b', label='Accuracy per fold')
    ax.axhline(y=cv_acc_scores.mean(), color='#22c55e', linestyle='--', label=f'Mean = {cv_acc_scores.mean():.1%}')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold", color='#c8c8ff')
    ax.set_ylabel("Accuracy", color='#c8c8ff')
    ax.set_title(f"{model_name} – 5-Fold CV Scores", color='#e0e0ff')
    ax.legend(facecolor='#1e1e3a', labelcolor='#c8c8ff')
    ax.tick_params(colors='#c8c8ff')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('#2a2a5a')
    ax.spines['left'].set_color('#2a2a5a')
    st.pyplot(fig)
    plt.close()
    
    # --- Recommendation Box (styled like STRATEGY_INFO) ---
    if gap > 0.05:
        box_color = '#ef4444'
        icon = '⚠️'
        title = 'Potential Overfitting Detected'
        message = f"""
        Train accuracy ({train_acc:.1%}) is **{gap:.1%} higher** than test accuracy ({test_acc:.1%}).  
        The model may be memorising the training data.
        """
    elif cv_acc_scores.std() > 0.03:
        box_color = '#f59e0b'
        icon = 'ℹ️'
        title = 'Moderate Instability'
        message = f"""
        Cross‑validation accuracy varies by {cv_acc_scores.std():.1%}.  
        The model's performance depends on the data split.
        """
    else:
        box_color = '#22c55e'
        icon = '✅'
        title = 'Good Generalisation'
        message = f"""
        Train‑test gap is only {gap:.1%} and cross‑validation scores are stable (std = {cv_acc_scores.std():.1%}).  
        The model is not overfitting and should perform well on unseen data.
        """
    
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.07); border-left:3px solid {box_color};
                border-radius:0 8px 8px 0; padding:12px 14px; margin-top:16px;'>
        <span style='color:#e0e0ff; font-weight:700;'>{icon} {title}</span><br>
        <span style='color:#9999cc; font-size:12px;'>{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# 🔍 OVERFITTING & CROSS-VALIDATION (Emotion Model)
# ──────────────────────────────────────────
def _render_overfitting_analysis_emotion(pipeline, X_train, y_train, X_test, y_test, X_full, y_full, model_name):
    """
    Computes train/test gap and 5-fold cross-validation for the emotion pipeline.
    Styling matches the stress analysis (bento cards, dark plot, recommendation box).
    """
    # Train accuracy
    train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    gap = train_acc - test_acc

    # Cross-validation on full dataset (using the pipeline)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc_scores = cross_val_score(pipeline, X_full, y_full, cv=cv, scoring='accuracy')
    cv_f1_scores  = cross_val_score(pipeline, X_full, y_full, cv=cv, scoring='f1_weighted')

    # --- Metric Cards (bento-grid) ---
    st.markdown("""
    <div class="bento-grid" style="margin-top:0px; margin-bottom:20px;">
        <div class="bento-card">
            <div class="bento-label">📈 Train Accuracy</div>
            <div class="bento-value">{:.1%}</div>
            <div class="bento-sub">on training split</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">📉 Test Accuracy</div>
            <div class="bento-value">{:.1%}</div>
            <div class="bento-sub">gap = {:.1%}</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">🔁 CV Accuracy</div>
            <div class="bento-value">{:.1%} ± {:.1%}</div>
            <div class="bento-sub">5-fold stratified</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">🎯 CV F1 (weighted)</div>
            <div class="bento-value">{:.1%} ± {:.1%}</div>
            <div class="bento-sub">5-fold stratified</div>
        </div>
    </div>
    """.format(train_acc, test_acc, gap, cv_acc_scores.mean(), cv_acc_scores.std(), cv_f1_scores.mean(), cv_f1_scores.std()), unsafe_allow_html=True)

    # --- Plot with dark theme ---
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#111128')
    ax.set_facecolor('#111128')
    ax.plot(range(1, 6), cv_acc_scores, 'o-', color='#f59e0b', label='Accuracy per fold')
    ax.axhline(y=cv_acc_scores.mean(), color='#22c55e', linestyle='--', label=f'Mean = {cv_acc_scores.mean():.1%}')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold", color='#c8c8ff')
    ax.set_ylabel("Accuracy", color='#c8c8ff')
    ax.set_title(f"{model_name} – Emotion Model 5-Fold CV Scores", color='#e0e0ff')
    ax.legend(facecolor='#1e1e3a', labelcolor='#c8c8ff')
    ax.tick_params(colors='#c8c8ff')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('#2a2a5a')
    ax.spines['left'].set_color('#2a2a5a')
    st.pyplot(fig)
    plt.close()

    # --- Recommendation Box ---
    if gap > 0.05:
        box_color = '#ef4444'
        icon = '⚠️'
        title = 'Potential Overfitting Detected'
        message = f"""
        Train accuracy ({train_acc:.1%}) is **{gap:.1%} higher** than test accuracy ({test_acc:.1%}).  
        The model may be memorising the training data.
        """
    elif cv_acc_scores.std() > 0.03:
        box_color = '#f59e0b'
        icon = 'ℹ️'
        title = 'Moderate Instability'
        message = f"""
        Cross‑validation accuracy varies by {cv_acc_scores.std():.1%}.  
        The model's performance depends on the data split.
        """
    else:
        box_color = '#22c55e'
        icon = '✅'
        title = 'Good Generalisation'
        message = f"""
        Train‑test gap is only {gap:.1%} and cross‑validation scores are stable (std = {cv_acc_scores.std():.1%}).  
        The model is not overfitting and should perform well on unseen emotion data.
        """
    
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.07); border-left:3px solid {box_color};
                border-radius:0 8px 8px 0; padding:12px 14px; margin-top:16px;'>
        <span style='color:#e0e0ff; font-weight:700;'>{icon} {title}</span><br>
        <span style='color:#9999cc; font-size:12px;'>{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# HELPER RENDERS
# ──────────────────────────────────────────
def _render_metric_cards(acc, prec, rec, f1, subtitle):
    st.markdown(f"""
    <div class="bento-grid" style="margin-top:20px;">
        <div class="bento-card">
            <div class="bento-label">Accuracy</div>
            <div class="bento-value">{acc:.1%}</div>
            <div class="bento-sub">{subtitle}</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">Precision</div>
            <div class="bento-value">{prec:.1%}</div>
            <div class="bento-sub">weighted avg</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">Recall</div>
            <div class="bento-value">{rec:.1%}</div>
            <div class="bento-sub">weighted avg</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">F1-Score</div>
            <div class="bento-value">{f1:.1%}</div>
            <div class="bento-sub">weighted avg</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_tfidf_analysis(tfidf, X_vec):
    st.markdown("##### 🔍 TF-IDF Keyword Analysis")
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores  = np.asarray(X_vec.mean(axis=0)).ravel()
    top_idx       = tfidf_scores.argsort()[-15:][::-1]
    top_words     = [feature_names[i] for i in top_idx]
    top_scores    = [tfidf_scores[i]  for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_words[::-1], top_scores[::-1])
    ax.set_xlabel("Average TF-IDF Score")
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_feature_importance(stress_model, tfidf):
    st.markdown("##### 🧠 Top Influential Words")
    feature_names = tfidf.get_feature_names_out()
    try:
        coef = stress_model.coef_
        class_labels = {
            0: ("😌 Normal",     "#22c55e"),
            1: ("😥 Mild Stress","#f59e0b"),
            2: ("😫 High Stress","#ef4444"),
        }
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            with col:
                label_name, _ = class_labels[idx]
                top10  = np.argsort(coef[idx])[-10:]
                words  = [feature_names[i] for i in top10]
                scores = [coef[idx][i]     for i in top10]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(words, scores)
                ax.set_title(label_name)
                ax.spines[['top', 'right']].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
    except Exception as e:
        st.warning(f"Feature importance tidak tersedia: {e}")


def _render_confusion_matrix(y_true, y_pred, unique_labels, label_map):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 🎯 Confusion Matrix")
    cm    = confusion_matrix(y_true, y_pred)
    ticks = [label_map.get(k, str(k)) for k in sorted(unique_labels)]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="magma", cbar=True,
        xticklabels=ticks[:cm.shape[1]], yticklabels=ticks[:cm.shape[0]],
        linewidths=2, linecolor="#0f0f1f", square=True,
        annot_kws={"size": 16, "weight": "bold", "color": "white"}, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", color="#c8c8ff", labelpad=12)
    ax.set_ylabel("Actual Label",    fontsize=12, fontweight="bold", color="#c8c8ff", labelpad=12)
    ax.tick_params(axis='x', colors='#ffffff', labelsize=10)
    ax.tick_params(axis='y', colors='#ffffff', labelsize=10)
    ax.set_facecolor("#111128")
    fig.patch.set_facecolor("#111128")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_classification_report(y_true, y_pred):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 📋 Classification Report")

    report    = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3).reset_index()
    report_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]

    h1, h2, h3, h4, h5 = st.columns([2, 1, 1, 1, 1])
    for col, label in zip([h1, h2, h3, h4, h5], ["*Class*", "*Precision*", "*Recall*", "*F1-Score*", "*Support*"]):
        col.markdown(label)
    st.markdown("---")

    for _, row in report_df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
        c1.markdown(f"<span style='color:#c8c8ff;font-weight:700'>{row['Class']}</span>",       unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#818cf8;font-family:monospace'>{row['Precision']:.3f}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:#22c55e;font-family:monospace'>{row['Recall']:.3f}</span>",    unsafe_allow_html=True)
        c4.markdown(f"<span style='color:#f59e0b;font-family:monospace'>{row['F1-Score']:.3f}</span>",  unsafe_allow_html=True)
        c5.markdown(f"<span style='color:#ef4444;font-family:monospace'>{int(row['Support'])}</span>",  unsafe_allow_html=True)
        st.markdown("<div class='report-row'></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
