"""
models.py — ML model constructors and balancing strategies.

Hyperparameters tuned for generalization over raw accuracy:
  LR  C=0.1, saga, l2     → strong regularization, less vocab memorization
  NB  alpha=1.0            → standard Laplace, handles OOV words
  SVM C=0.1                → wide margin, less overfit on train vocabulary
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


def build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced",
            C=0.1, solver="saga", penalty="l2", random_state=42,
        )
    elif name == "Naive Bayes":
        return MultinomialNB(alpha=1.0)
    elif name == "Linear SVM":
        return LinearSVC(
            class_weight="balanced", max_iter=3000,
            C=0.1, random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


def apply_balancing(X_vec, y, strategy: str):
    """Call AFTER train/test split — never on full dataset."""
    if strategy == "Random Oversampling":
        sampler = RandomOverSampler(random_state=42)
    elif strategy == "SMOTE":
        k = max(1, min(3, int(np.bincount(y.astype(int)).min()) - 1))
        sampler = SMOTE(random_state=42, k_neighbors=k)
    elif strategy == "Random Undersampling":
        sampler = RandomUnderSampler(random_state=42)
    elif strategy == "SMOTETomek":
        sampler = SMOTETomek(random_state=42)
    else:
        return X_vec, y
    return sampler.fit_resample(X_vec, y)


STRATEGY_INFO = {
    "Random Oversampling": ("Menduplikasi sampel kelas minoritas secara acak.", "#6366f1"),
    "SMOTE":               ("Membuat sampel sintetis dari interpolasi kelas minoritas.", "#a78bfa"),
    "Random Undersampling":("Mengurangi sampel kelas mayoritas secara acak.", "#f59e0b"),
    "SMOTETomek":          ("SMOTE + Tomek Links untuk boundary yang lebih bersih.", "#22c55e"),
    "Tanpa Balancing":     ("Tidak ada penyeimbangan — gunakan jika data sudah balance.", "#ef4444"),
}
