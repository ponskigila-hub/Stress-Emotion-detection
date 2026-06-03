# 🧠 MindScan — NLP Stress Detection

A Streamlit web application that detects **emotions** and **stress levels** from Indonesian social media text using Natural Language Processing and Machine Learning.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [File Reference](#file-reference)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [App Pages](#app-pages)
- [ML Pipeline](#ml-pipeline)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Anti-Overfitting Strategy](#anti-overfitting-strategy)
- [Balancing Strategies](#balancing-strategies)
- [Confidence-Weighted Majority Voting](#confidence-weighted-majority-voting)

---

## Overview

MindScan analyzes Indonesian text (tweets, forum posts) and classifies it into:

- **Emotion** — `happy`, `sad`, `anger`, `fear`, `love`, `surprise`, `neutral`
- **Stress Level** — `0 = Normal`, `1 = Mild Stress`, `2 = High Stress`

It supports both single-text prediction and **bulk CSV clinical analysis**, where all posts from one user are analyzed together and a final mental health verdict is produced using confidence-weighted majority voting.

---

## Features

- 📊 **EDA** — Distribution charts, word clouds, and slang language analysis
- ⚙️ **Preprocessing** — Live tokenization and stemming visualizations, before/after comparisons
- 🤖 **Model Training** — Configurable model, TF-IDF, balancing, with overfitting detection and cross-validation score
- 🔮 **Prediction** — Single text analysis with confidence scores
- 📂 **Bulk Clinical Analysis** — Upload a CSV of one user's posts, get a clinical mental health verdict
- 🌙 **Dark UI** — Bento-style dark theme, horizontal top navigation bar

---

## Project Structure

```
Stress-Emotion-detection-main/
│
├── app.py                        # Entry point — routing, top navbar, data bootstrap
├── styles.py                     # All CSS styles and matplotlib dark theme
├── data_loader.py                # Dataset loading, caching, label/color constants
├── utils.py                      # Text cleaning, slang normalization, stopwords, stemming
├── models.py                     # Model factory, balancing strategies
├── requirements.txt              # Python dependencies
│
├── views/
│   ├── __init__.py
│   ├── home.py                   # 🏠 Home page
│   ├── eda.py                    # 📊 EDA page
│   ├── preprocessing.py          # ⚙️  Preprocessing page
│   ├── training.py               # 🤖 Model Training page
│   └── prediction.py             # 🔮 Prediction + 📂 Bulk Clinical Analysis
│
├── data/
│   ├── emotion_accuracy_training.csv
│   ├── ugm_fess_labeled.csv
│   └── slang_indo.csv
│
└── user_analysis_example/
    └── contoh_bulk_test.csv      # Sample CSV for bulk analysis testing
```

> **Note:** Page modules live in `views/` (not `pages/`) to prevent Streamlit's built-in multipage navigator from appearing. Navigation is handled via a custom **horizontal top navbar** in `app.py`.

---

## File Reference

### `app.py`
Main entry point. Responsibilities:
- Sets page config (title, icon, wide layout)
- Injects global styles and matplotlib theme
- Initializes session state: `emotion_model`, `stress_model`, `last_metrics`, `train_log`, `current_page`
- Loads and preprocesses both datasets (cached)
- Renders the **horizontal top navigation bar** using `st.columns`
- Routes to the correct view module based on `st.session_state.current_page`
- Renders sidebar dataset status cards

### `styles.py`
All CSS — no logic. Three string blocks:

| Block | Contents |
|---|---|
| `MAIN_CSS` | Layout, bento cards, buttons, tabs, hero banner, prediction panels, top navbar styles |
| `METRIC_CSS` | Styled `st.metric` cards with glowing top borders |
| `SIDEBAR_CSS` | Sidebar dataset status panel styling |

| Function | Purpose |
|---|---|
| `inject_styles()` | Injects MAIN_CSS + METRIC_CSS |
| `inject_sidebar_styles()` | Injects SIDEBAR_CSS |
| `set_matplotlib_theme()` | Dark rcParams for all plots |

### `data_loader.py`
Data I/O and shared constants.

| Item | Description |
|---|---|
| `load_data()` | Loads emotion and stress CSVs. `@st.cache_data`. |
| `load_slang_dict()` | Reads `slang_indo.csv` → `{slang: formal}`. Cached. |
| `STRESS_LABEL_MAP` | `{0: "Normal", 1: "Mild Stress", 2: "High Stress"}` |
| `STRESS_COLORS` | Hex colors per stress label |
| `EMOTION_LABEL_MAP` | Display names with emoji per emotion |
| `EMOTION_COLORS` | Hex colors per emotion |

### `utils.py`
Pure NLP utilities — no Streamlit dependency.

| Item | Description |
|---|---|
| `stemmer` | Sastrawi stemmer, initialized once at module level |
| `KEEP_WORDS` | Emotionally significant words exempt from stopword removal |
| `remove_repeated_char(text)` | Collapses 3+ repeated chars to 2 |
| `normalize_slang(text, slang_dict)` | Replaces slang with formal equivalents |
| `remove_stopwords(text)` | Sastrawi stopword removal, skipping `KEEP_WORDS` |
| `clean_text(text, slang_dict)` | Full pipeline: lowercase → URLs → @/# → alpha → repeats → slang → stopwords → stem |
| `detect_slang_words(text, slang_words)` | Returns list of slang tokens in text |

**Why `KEEP_WORDS` matters:** Sastrawi removes words like `tidak`, `bukan`, `sangat` which are critical for stress detection. `KEEP_WORDS` preserves negations, intensifiers, and emotional vocabulary.

### `models.py`
Model construction and resampling.

| Item | Description |
|---|---|
| `build_model(name)` | Returns untrained classifier with generalization-tuned hyperparameters |
| `apply_balancing(X_vec, y, strategy)` | Resamples — call on training split only |
| `STRATEGY_INFO` | `{strategy: (description, color)}` for UI |

**Hyperparameter rationale:**

| Model | Param | Value | Reason |
|---|---|---|---|
| Logistic Regression | `C` | `0.1` | Strong L2 → less vocab memorization |
| Logistic Regression | `solver` | `saga` | Efficient multiclass + L2 |
| Naive Bayes | `alpha` | `1.0` | Standard Laplace → handles unseen words |
| Linear SVM | `C` | `0.1` | Wide margin → less noise sensitivity |

### `views/home.py`
Landing page: hero banner, 4 bento metric cards, usage flow guide, stress label bar chart.

**Function:** `render(emotion_df, stress_df)`

### `views/eda.py`
Four-tab EDA page.

**Function:** `render(emotion_df, stress_df, slang_words)`

| Tab | Content |
|---|---|
| Emosi | Data preview, bar + pie chart of emotion distribution |
| Stres | Data preview, stress label bar chart, text length histogram |
| WordCloud | Configurable word cloud by label, slang word cloud |
| Slang Analysis | Slang stats bento cards, top slang charts, before/after normalization |

### `views/preprocessing.py`
Interactive preprocessing demo.

**Function:** `render(emotion_df, stress_df, slang_dict)`

- Live tokenization (badge per token)
- Live stemming table (original vs stemmed)
- 5 cleaning step cards
- Before/after comparison from emotion dataset
- Manual tester with removed-token badges
- Avg text length and vocabulary size stats

### `views/training.py`
Full training workflow with anti-overfit improvements.

**Function:** `render(emotion_df, stress_df)`

**Correct training order:**
1. Split train/test with `stratify=True`
2. Fit TF-IDF on training set only (`sublinear_tf=True`, `min_df=2`, `max_df=0.95`)
3. Apply balancing to training vectors only
4. Train classifier
5. Evaluate on untouched test set
6. Run 5-fold cross-validation → show CV F1 ± std

Shows overfitting warning if `train_acc − test_acc > 15%`.

### `views/prediction.py`
Two-tab prediction page.

**Function:** `render(slang_dict)`

**Tab 1 — Single Text:** Emotion + stress prediction with per-class confidence bars.

**Tab 2 — Bulk Clinical Analysis (👤 Analisis User Medsos):**
- Upload CSV of all posts from one user
- Auto-detects text column (`text`, `tweet`, `post`, `content`, or first column)
- Confidence-weighted majority voting across all posts
- Clinical verdict banner: dominant stress, dominant emotion, interpretation, referral recommendation
- Timeline scatter chart of per-post confidence scores
- Per-post detail table
- Export: full CSV + one-row clinical summary CSV

---

## Data Requirements

Place these three files in the `data/` folder:

| File | Required Columns | Description |
|---|---|---|
| `emotion_accuracy_training.csv` | `tweet`, `label` | Tweet text + emotion label string |
| `ugm_fess_labeled.csv` | `full_text`, `*label*` | Post text + numeric stress label (0/1/2) |
| `slang_indo.csv` | col 0 = slang, col 1 = formal | Indonesian slang normalization dictionary |

The stress label column is auto-detected — any column whose name contains `"label"` is used.

---

## Installation

**Python 3.8+ recommended.**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn wordcloud PySastrawi
```

---

## Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## App Pages

Navigation is via the **horizontal top navbar** (not sidebar).

| Page | Purpose |
|---|---|
| 🏠 Home | Overview, dataset summary, usage guide |
| 📊 EDA | Explore distributions, word clouds, slang analysis |
| ⚙️ Preprocessing | See and test the cleaning pipeline |
| 🤖 Model Training | Train, evaluate, detect overfitting |
| 🔮 Prediction | Single text OR bulk CSV clinical analysis |

> Train a model on the **Model Training** page first. Models are stored in session state and reset when the browser tab closes.

---

## ML Pipeline

```
Raw Text
   │
   ▼
clean_text()
   │  lowercase → strip URLs/@/# → keep alpha
   │  → collapse repeats → normalize slang
   │  → remove stopwords (preserve KEEP_WORDS) → stem
   ▼
Train/Test Split  (stratify=True)
   │
   ├─ Training Set ─────────────────────────────────────┐
   │     TF-IDF fit_transform()                         │
   │       sublinear_tf=True, min_df=2, max_df=0.95     │
   │     Balancing (training vectors only)              │
   │     Classifier.fit()                               │
   │       LR C=0.1 / NB alpha=1.0 / SVM C=0.1         │
   │                                                    │
   └─ Test Set ─────────────────────────────────────────┤
         TF-IDF transform() only                        │
         Classifier.predict() → Metrics ◄───────────────┘
                │
                ▼
         5-Fold CV on training set → CV F1 ± std
```

---

## Text Preprocessing Pipeline

| Step | What it does | Example |
|---|---|---|
| Lowercase | All to lowercase | `"Capek BANGET"` → `"capek banget"` |
| Remove URLs | Strip links | `"cek http://t.co/x"` → `"cek"` |
| Remove @/# | Strip mentions/hashtags | `"@teman #stress"` → `""` |
| Remove non-alpha | Keep letters + spaces | `"capek!!!123"` → `"capek"` |
| Collapse repeats | Max 2 identical consecutive chars | `"capeeeeek"` → `"capee"` |
| Normalize slang | Replace with formal word | `"gw"` → `"saya"` |
| Remove stopwords | Sastrawi, skip `KEEP_WORDS` | `"dan aku sangat capek"` → `"aku sangat capek"` |
| Stem | Sastrawi stemmer | `"melelahkan"` → `"lelah"` |

---

## Anti-Overfitting Strategy

| Problem | Old behavior | Fixed behavior |
|---|---|---|
| Data leakage | Balancing before split | Split first, balance training only |
| TF-IDF leakage | `fit_transform` on full data | `fit` on train, `transform` on test |
| No stratify | Random split skews class ratios | `stratify=y` preserves distribution |
| High C | C=1.0 → memorizes training vocab | C=0.1 → stronger regularization |
| No sublinear_tf | Dominant terms overwhelm features | `sublinear_tf=True` → log scaling |
| No min_df | Rare noise words as features | `min_df=2` drops single-doc tokens |
| No stopwords | Common words fill feature space | Sastrawi stopword removal |
| No stemming | Same word = multiple features | All variants stem to same root |
| No CV | No way to detect overfitting | 5-fold CV F1 shown after training |
| No gap alert | User unaware of train-test gap | Warning if gap > 15% |

---

## Balancing Strategies

Applied **only to the training split**, never to the test set.

| Strategy | Method | Best for |
|---|---|---|
| Random Oversampling | Duplicates minority samples | Quick baseline |
| SMOTE | Synthetic samples via interpolation | Medium datasets |
| Random Undersampling | Removes majority samples | Large datasets |
| SMOTETomek | SMOTE + Tomek Links for clean boundaries | Best quality |
| Tanpa Balancing | No resampling | Already balanced data |

---

## Confidence-Weighted Majority Voting

Used in Bulk Clinical Analysis. More robust than simple majority voting.

| Step | Detail |
|---|---|
| Per-post prediction | Each post → label + full probability array |
| Accumulate | Sum confidence scores per class across all posts |
| Average | Divide by total posts → avg confidence per class |
| Verdict | Class with highest average confidence = final label |

```
Example — 3 posts:
  Post 1 → Normal: 0.80  Mild: 0.15  High: 0.05
  Post 2 → Normal: 0.30  Mild: 0.60  High: 0.10
  Post 3 → Normal: 0.20  Mild: 0.70  High: 0.10
  ──────────────────────────────────────────────
  Avg    → Normal: 0.43  Mild: 0.48  High: 0.08
  Verdict → Mild Stress  (majority vote = tie; confidence breaks it)
```

### Bulk CSV Format

```
text
Hari ini gue ngerasa capek banget
Udah 3 hari ga bisa tidur, kepala pusing
Seneng banget hari ini bisa ketemu temen lama
```

Accepted column names: `text`, `tweet`, `post`, `content`, `kalimat`, `teks`. Falls back to first column if none match. A sample file is provided at `user_analysis_example/contoh_bulk_test.csv`.
