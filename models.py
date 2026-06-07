# models.py (updated)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, penalty='l2', solver='saga')
    elif name == "Naive Bayes":
        return MultinomialNB(alpha=0.1)
    elif name == "Linear SVM":
        return LinearSVC(class_weight="balanced", max_iter=5000, C=1.0)
    else:
        raise ValueError(f"Unknown model: {name}")


# ==========================================
# BALANCING STRATEGIES
# ==========================================
def apply_balancing(X_vec, y, strategy: str):
    """
    Resample (X_vec, y) using the chosen strategy.
    Returns (X_resampled, y_resampled).
    """
    if strategy == "Random Oversampling":
        sampler = RandomOverSampler(random_state=42)

    elif strategy == "SMOTE":
        k = min(3, int(np.bincount(y.astype(int)).min()) - 1)
        sampler = SMOTE(random_state=42, k_neighbors=k)

    elif strategy == "Random Undersampling":
        sampler = RandomUnderSampler(random_state=42)

    elif strategy == "SMOTETomek":
        sampler = SMOTETomek(random_state=42)

    else:  # "Tanpa Balancing"
        return X_vec, y

    return sampler.fit_resample(X_vec, y)


# ==========================================
# STRATEGY DESCRIPTIONS (for UI)
# ==========================================
STRATEGY_INFO = {
    "Random Oversampling": (
        "Menduplikasi sampel kelas minoritas secara acak.", "#6366f1"
    ),
    "SMOTE": (
        "Membuat sampel sintetis baru dari interpolasi kelas minoritas.", "#a78bfa"
    ),
    "Random Undersampling": (
        "Mengurangi sampel kelas mayoritas secara acak.", "#f59e0b"
    ),
    "SMOTETomek": (
        "Kombinasi SMOTE + Tomek Links untuk boundary yang bersih.", "#22c55e"
    ),
    "Tanpa Balancing": (
        "Tidak ada penyeimbangan — gunakan jika data sudah balance.", "#ef4444"
    ),
}
