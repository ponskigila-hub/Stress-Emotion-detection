# 🧠 MindScan — NLP Stress Detection

A Streamlit web application that detects **emotions** and **stress levels** from Indonesian social media text using Natural Language Processing and Machine Learning.

---

## Auhtors
BINUS University Students:
1. Jonathan Raffael - 2802455275
2. ⁠Darren Star Limantoro - 2802461422
3. Steven Hosea - 2802453591
4. Albertus Adrian - 2802451876
5. Nicholas Driyadis Tjoe - 2802461321

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
- [Balancing Strategies](#balancing-strategies)

---

## Overview

MindScan analyzes Indonesian text (tweets, forum posts) and classifies it into:

- **Emotion** — `happy`, `sad`, `anger`, `fear`, `love`, `surprise`, `neutral`
- **Stress Level** — `0 = Normal`, `1 = Mild Stress`, `2 = High Stress`

It uses TF-IDF vectorization combined with classical ML classifiers (Logistic Regression, Naive Bayes, Linear SVM), and supports multiple class-imbalance handling strategies.

---

## Features

- 📊 **EDA** — Distribution charts, word clouds, and slang language analysis
- ⚙️ **Preprocessing** — Live tokenization and stemming visualizations, before/after comparisons
- 🤖 **Model Training** — Configurable model, TF-IDF features, test split, and balancing strategy with real-time progress tracking
- 🔮 **Prediction** — Input any text and get instant emotion + stress detection with confidence scores
- 🌙 **Dark UI** — Bento-style dark theme with custom CSS throughout

---

## Project Structure

```
mindscan/
│
├── app.py                  # Entry point — page routing, sidebar, data bootstrap
├── styles.py               # All CSS styles and matplotlib dark theme
├── data_loader.py          # Dataset loading, caching, label/color constants
├── utils.py                # Text cleaning, slang normalization, stemming
├── models.py               # Model factory, balancing strategies
│
├── views/
│   ├── __init__.py
│   ├── home.py             # 🏠 Home page
│   ├── eda.py              # 📊 EDA page
│   ├── preprocessing.py    # ⚙️  Preprocessing page
│   ├── training.py         # 🤖 Model Training page
│   └── prediction.py       # 🔮 Prediction page
│
└── data/                   # ← you provide this folder
    ├── emotion_accuracy_training.csv
    ├── ugm_fess_labeled.csv
    └── slang_indo.csv
```

---

## File Reference

### `app.py`
The main Streamlit entry point. Responsibilities:
- Sets page config (title, icon, layout)
- Injects global styles and matplotlib theme
- Initializes session state keys: `emotion_model`, `stress_model`, `last_metrics`, `train_log`
- Loads and preprocesses both datasets (cached)
- Renders the sidebar with navigation and dataset status cards
- Routes to the correct page module based on sidebar selection

### `styles.py`
Contains all visual styling. Nothing functional — purely CSS and theme config.

| Function | Purpose |
|---|---|
| `inject_styles()` | Injects main CSS + metric card CSS into the app |
| `inject_sidebar_styles()` | Injects sidebar radio button CSS (called inside sidebar context) |
| `set_matplotlib_theme()` | Sets dark background/color rcParams for all matplotlib plots |

Three CSS blocks are defined as module-level strings:
- `MAIN_CSS` — layout, bento cards, buttons, tabs, hero banner, prediction result panels
- `METRIC_CSS` — styled `st.metric` cards with glowing top borders
- `SIDEBAR_CSS` — custom radio buttons (no bullet, centered text, consistent box size)

### `data_loader.py`
Handles all data I/O and shared constants.

| Item | Description |
|---|---|
| `load_data()` | Loads and normalizes the emotion and stress CSVs. Cached with `@st.cache_data`. |
| `load_slang_dict()` | Reads `slang_indo.csv` into a `{slang: formal}` dict. Cached. |
| `STRESS_LABEL_MAP` | `{0: "Normal", 1: "Mild Stress", 2: "High Stress"}` |
| `STRESS_COLORS` | Hex colors for each stress label (green / amber / red) |
| `EMOTION_LABEL_MAP` | Display names with emoji for each emotion class |
| `EMOTION_COLORS` | Hex colors for each emotion class |

### `utils.py`
Pure NLP utility functions with no Streamlit dependency.

| Function | Description |
|---|---|
| `stemmer` | Module-level Sastrawi stemmer instance (initialized once) |
| `remove_repeated_char(text)` | Collapses 3+ repeated chars to 2, e.g. `capeeek → capee` |
| `normalize_slang(text, slang_dict)` | Replaces slang tokens with their formal equivalents |
| `clean_text(text, slang_dict)` | Full pipeline: lowercase → strip URLs → strip @/# → keep alpha → collapse repeats → strip whitespace → normalize slang |
| `detect_slang_words(text, slang_words)` | Returns list of slang tokens found in text |

### `models.py`
ML model construction and resampling logic.

| Item | Description |
|---|---|
| `build_model(name)` | Returns an untrained sklearn classifier by name |
| `apply_balancing(X_vec, y, strategy)` | Resamples feature matrix and labels using the chosen strategy |
| `STRATEGY_INFO` | Dict mapping strategy names to `(description, hex_color)` tuples for UI display |

Supported models: `Logistic Regression`, `Naive Bayes`, `Linear SVM`

### `views/home.py`
Renders the landing page. Shows the hero banner, four summary metric bento cards (total emotion data, total stress data, normal count, stress count), a usage flow guide, and a stress label distribution bar chart.

**Function:** `render(emotion_df, stress_df)`

### `views/eda.py`
Four-tab exploratory data analysis page.

**Function:** `render(emotion_df, stress_df, slang_words)`

| Tab | Content |
|---|---|
| Emosi | Data preview table, horizontal bar chart + pie chart of emotion class distribution |
| Stres | Data preview table, stress label bar chart, text length histogram by stress level |
| WordCloud | Configurable word cloud (by dataset/label), slang word cloud |
| Slang Analysis | Slang stats bento cards, top slang bar + pie charts, before/after normalization cards |

### `views/preprocessing.py`
Demonstrates the preprocessing pipeline interactively.

**Function:** `render(emotion_df, stress_df, slang_dict)`

- Live tokenization visualizer — renders each token as a styled badge
- Live stemming table — shows original vs stemmed tokens side by side
- Five step cards explaining the cleaning stages
- Sample before/after comparison from the emotion dataset
- Manual preprocessing tester with token-removal badges
- Stats panel: average text length (emotion), average text length (stress), vocabulary size

### `views/training.py`
Full model training workflow.

**Function:** `render(emotion_df, stress_df)`

Trains **two models simultaneously**:
1. **Emotion model** — a `Pipeline(TF-IDF + classifier)` trained on `emotion_df`
2. **Stress model** — a standalone classifier trained on TF-IDF vectors of `stress_df`, after optional resampling

After training, displays:
- Metric bento cards (Accuracy, Precision, Recall, F1) for both models
- TF-IDF keyword importance bar chart
- Top influential words per class (Logistic Regression and SVM only)
- Confusion matrices (heatmap)
- Classification reports (per-class precision/recall/F1/support)

Trained models and metrics are saved to `st.session_state` for use by the Prediction page.

### `views/prediction.py`
Two-tab prediction page — single text and bulk CSV clinical analysis.

**Function:** `render(slang_dict)`

**Tab 1 — Analisis Teks Tunggal:** Single post analysis with emotion + stress prediction and per-class confidence bars.

**Tab 2 — Analisis CSV Bulk (Clinical):**
- Upload a CSV of all social media posts from one user (columns: `text`, `tweet`, `post`, `content`, or first column)
- Runs confidence-weighted majority voting across every post
- Renders a clinical verdict banner: dominant stress level, dominant emotion, clinical interpretation, and referral recommendation
- Timeline scatter chart of confidence scores across posts
- Full per-post detail table (stress label, stress conf, emotion, emotion conf)
- Export: full detail CSV + clinical summary CSV

---

## Data Requirements

Place these three files in a `data/` folder next to `app.py`:

| File | Required Columns | Description |
|---|---|---|
| `emotion_accuracy_training.csv` | `tweet`, `label` | Tweet text + emotion label string |
| `ugm_fess_labeled.csv` | `full_text`, `*label*` | Post text + numeric stress label (0/1/2) |
| `slang_indo.csv` | col 0 = slang, col 1 = formal | Indonesian slang normalization dictionary |

The stress label column is detected automatically — any column whose name contains `"label"` (case-insensitive) is used.

---

## Installation

**Python 3.8+ recommended.**

Install all dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn wordcloud PySastrawi
```

Or create a `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
wordcloud
PySastrawi
```

Then run:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
cd mindscan
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## App Pages

Navigate using the sidebar menu:

| Page | Purpose |
|---|---|
| 🏠 Home | Overview, dataset summary, usage guide |
| 📊 EDA | Explore data distributions, word clouds, slang analysis |
| ⚙️ Preprocessing | See and test the text cleaning pipeline |
| 🤖 Model Training | Configure and train ML models, view evaluation metrics |
| 🔮 Prediction | Detect emotion and stress from any text input |

> **Note:** You must train a model on the **Model Training** page before the **Prediction** page will work. Models are held in session state and reset when the browser tab is closed.

---

## ML Pipeline

```
Raw Text
   │
   ▼
clean_text()          ← lowercase, remove URLs/@/#/symbols,
   │                    collapse repeated chars, normalize slang
   ▼
TF-IDF Vectorizer     ← max_features configurable (1k–20k),
   │                    bigrams enabled (ngram_range=(1,2))
   ▼
Balancing Strategy    ← optional resampling on training split only
   │
   ▼
Classifier            ← Logistic Regression / Naive Bayes / Linear SVM
   │
   ▼
Predicted Label       ← stress: 0/1/2   emotion: string class
```

Both models share the same TF-IDF feature space for stress detection. The emotion model uses a full sklearn `Pipeline` so vectorization and classification happen in a single `.predict()` call.

---

## Text Preprocessing Pipeline

Each text goes through these steps in order inside `clean_text()`:

| Step | What it does | Example |
|---|---|---|
| Lowercase | Converts all characters to lowercase | `"Capek BANGET"` → `"capek banget"` |
| Remove URLs | Strips `http://`, `https://`, `www.` links | `"cek http://t.co/x"` → `"cek "` |
| Remove @/# | Strips mentions and hashtags | `"@teman #stress"` → `" "` |
| Remove non-alpha | Keeps only letters and spaces | `"capek!!!123"` → `"capek"` |
| Collapse repeats | Max 2 consecutive identical chars | `"capeeeeek"` → `"capee"` |
| Strip whitespace | Normalizes multiple spaces | `"capek  banget"` → `"capek banget"` |
| Normalize slang | Replaces slang with formal words | `"gw"` → `"saya"` |

---

## Balancing Strategies

Class imbalance is handled before fitting the stress classifier. Resampling is applied **only to the training split**, never to the test split.

| Strategy | Method | Best for |
|---|---|---|
| Random Oversampling | Duplicates minority class samples randomly | Quick fix, small datasets |
| SMOTE | Generates synthetic minority samples via interpolation | Medium datasets, smooth boundaries |
| Random Undersampling | Removes majority class samples randomly | Large datasets where data loss is acceptable |
| SMOTETomek | SMOTE + Tomek Links to clean boundary noise | Best quality, slower |
| Tanpa Balancing | No resampling | Already balanced data |

---

## Confidence-Weighted Majority Voting (Bulk Analysis)

This is the core clinical decision-support logic used in the **CSV Bulk Analysis** tab. Unlike simple majority voting (most frequent label wins), this method weights each prediction by the model's confidence.

| Step | Detail |
|---|---|
| Per-post prediction | Each post → label + full probability array (one score per class) |
| Accumulate | Sum confidence scores per class across all posts |
| Average | Divide by total post count → average confidence per class |
| Verdict | Class with highest average confidence = final label |

A post predicted as "High Stress" with 95% confidence contributes far more to the conclusion than one predicted at 51%. This makes the system robust to noisy or ambiguous individual posts.

```
Example — 3 posts:
  Post 1 → Normal: 0.80  Mild: 0.15  High: 0.05
  Post 2 → Normal: 0.30  Mild: 0.60  High: 0.10
  Post 3 → Normal: 0.20  Mild: 0.70  High: 0.10
  ──────────────────────────────────────────────
  Sum    → Normal: 1.30  Mild: 1.45  High: 0.25
  Avg    → Normal: 0.43  Mild: 0.48  High: 0.08
  Verdict → Mild Stress  (majority vote alone = tie Normal/Mild)
```

### CSV Format Requirements

The uploaded file should look like this:

```
text
Hari ini gue ngerasa capek banget, tugas numpuk ga kelar-kelar
Seneng banget hari ini bisa ketemu temen lama
Udah 3 hari ga bisa tidur, kepala pusing terus
...
```

Accepted column names: `text`, `tweet`, `post`, `content`, `kalimat`, `teks`. If none match, the first column is used automatically.

### Clinical Output

After analysis, the system produces:
- **Dominant stress level** with average confidence score
- **Dominant emotion** across all posts
- **Clinical interpretation** and referral recommendation
- **Timeline chart** showing confidence per post over time
- **Exportable CSVs**: full per-post detail + one-row clinical summary

# 🧠 MindScan — NLP Stress Detection

A Streamlit web application that detects **emotions** and **stress levels** from Indonesian social media text using Natural Language Processing and Machine Learning.

## Authors
BINUS University students:
1. Albertus Adrian
2. Darren Star Limantoro
3. Jonathan Raffael  
4. Nicholas Driyadis Tjoe
5. Steven Hosea 

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
- [Balancing Strategies](#balancing-strategies)

---

## Overview

MindScan analyzes Indonesian text (tweets, forum posts) and classifies it into:

- **Emotion** — `happy`, `sad`, `anger`, `fear`, `love`, `surprise`, `neutral`
- **Stress Level** — `0 = Normal`, `1 = Mild Stress`, `2 = High Stress`

It uses TF-IDF vectorization combined with classical ML classifiers (Logistic Regression, Naive Bayes, Linear SVM), and supports multiple class-imbalance handling strategies.

---

## Features

- 📊 **EDA** — Distribution charts, word clouds, and slang language analysis
- ⚙️ **Preprocessing** — Live tokenization and stemming visualizations, before/after comparisons
- 🤖 **Model Training** — Configurable model, TF-IDF features, test split, and balancing strategy with real-time progress tracking
- 🔮 **Prediction** — Input any text and get instant emotion + stress detection with confidence scores
- 🌙 **Dark UI** — Bento-style dark theme with custom CSS throughout

---

## Project Structure

```
mindscan/
│
├── app.py                  # Entry point — page routing, sidebar, data bootstrap
├── styles.py               # All CSS styles and matplotlib dark theme
├── data_loader.py          # Dataset loading, caching, label/color constants
├── utils.py                # Text cleaning, slang normalization, stemming
├── models.py               # Model factory, balancing strategies
│
├── pages/
│   ├── __init__.py
│   ├── home.py             # 🏠 Home page
│   ├── eda.py              # 📊 EDA page
│   ├── preprocessing.py    # ⚙️  Preprocessing page
│   ├── training.py         # 🤖 Model Training page
│   └── prediction.py       # 🔮 Prediction page
│
└── data/                   
    ├── emotion_accuracy_training.csv # Emotion Data
    ├── ugm_fess_labeled.csv # Stress Data
    └── slang_indo.csv # Slang Data Indonesia
```

---

## File Reference

### `app.py`
The main Streamlit entry point. Responsibilities:
- Sets page config (title, icon, layout)
- Injects global styles and matplotlib theme
- Initializes session state keys: `emotion_model`, `stress_model`, `last_metrics`, `train_log`
- Loads and preprocesses both datasets (cached)
- Renders the sidebar with navigation and dataset status cards
- Routes to the correct page module based on sidebar selection

### `styles.py`
Contains all visual styling. Nothing functional — purely CSS and theme config.

| Function | Purpose |
|---|---|
| `inject_styles()` | Injects main CSS + metric card CSS into the app |
| `inject_sidebar_styles()` | Injects sidebar radio button CSS (called inside sidebar context) |
| `set_matplotlib_theme()` | Sets dark background/color rcParams for all matplotlib plots |

Three CSS blocks are defined as module-level strings:
- `MAIN_CSS` — layout, bento cards, buttons, tabs, hero banner, prediction result panels
- `METRIC_CSS` — styled `st.metric` cards with glowing top borders
- `SIDEBAR_CSS` — custom radio buttons (no bullet, centered text, consistent box size)

### `data_loader.py`
Handles all data I/O and shared constants.

| Item | Description |
|---|---|
| `load_data()` | Loads and normalizes the emotion and stress CSVs. Cached with `@st.cache_data`. |
| `load_slang_dict()` | Reads `slang_indo.csv` into a `{slang: formal}` dict. Cached. |
| `STRESS_LABEL_MAP` | `{0: "Normal", 1: "Mild Stress", 2: "High Stress"}` |
| `STRESS_COLORS` | Hex colors for each stress label (green / amber / red) |
| `EMOTION_LABEL_MAP` | Display names with emoji for each emotion class |
| `EMOTION_COLORS` | Hex colors for each emotion class |

### `utils.py`
Pure NLP utility functions with no Streamlit dependency.

| Function | Description |
|---|---|
| `stemmer` | Module-level Sastrawi stemmer instance (initialized once) |
| `remove_repeated_char(text)` | Collapses 3+ repeated chars to 2, e.g. `capeeek → capee` |
| `normalize_slang(text, slang_dict)` | Replaces slang tokens with their formal equivalents |
| `clean_text(text, slang_dict)` | Full pipeline: lowercase → strip URLs → strip @/# → keep alpha → collapse repeats → strip whitespace → normalize slang |
| `detect_slang_words(text, slang_words)` | Returns list of slang tokens found in text |

### `models.py`
ML model construction and resampling logic.

| Item | Description |
|---|---|
| `build_model(name)` | Returns an untrained sklearn classifier by name |
| `apply_balancing(X_vec, y, strategy)` | Resamples feature matrix and labels using the chosen strategy |
| `STRATEGY_INFO` | Dict mapping strategy names to `(description, hex_color)` tuples for UI display |

Supported models: `Logistic Regression`, `Naive Bayes`, `Linear SVM`

### `pages/home.py`
Renders the landing page. Shows the hero banner, four summary metric bento cards (total emotion data, total stress data, normal count, stress count), a usage flow guide, and a stress label distribution bar chart.

**Function:** `render(emotion_df, stress_df)`

### `pages/eda.py`
Four-tab exploratory data analysis page.

**Function:** `render(emotion_df, stress_df, slang_words)`

| Tab | Content |
|---|---|
| Emosi | Data preview table, horizontal bar chart + pie chart of emotion class distribution |
| Stres | Data preview table, stress label bar chart, text length histogram by stress level |
| WordCloud | Configurable word cloud (by dataset/label), slang word cloud |
| Slang Analysis | Slang stats bento cards, top slang bar + pie charts, before/after normalization cards |

### `pages/preprocessing.py`
Demonstrates the preprocessing pipeline interactively.

**Function:** `render(emotion_df, stress_df, slang_dict)`

- Live tokenization visualizer — renders each token as a styled badge
- Live stemming table — shows original vs stemmed tokens side by side
- Five step cards explaining the cleaning stages
- Sample before/after comparison from the emotion dataset
- Manual preprocessing tester with token-removal badges
- Stats panel: average text length (emotion), average text length (stress), vocabulary size

### `pages/training.py`
Full model training workflow.

**Function:** `render(emotion_df, stress_df)`

Trains **two models simultaneously**:
1. **Emotion model** — a `Pipeline(TF-IDF + classifier)` trained on `emotion_df`
2. **Stress model** — a standalone classifier trained on TF-IDF vectors of `stress_df`, after optional resampling

After training, displays:
- Metric bento cards (Accuracy, Precision, Recall, F1) for both models
- TF-IDF keyword importance bar chart
- Top influential words per class (Logistic Regression and SVM only)
- Confusion matrices (heatmap)
- Classification reports (per-class precision/recall/F1/support)

Trained models and metrics are saved to `st.session_state` for use by the Prediction page.

### `pages/prediction.py`
Real-time text analysis page.

**Function:** `render(slang_dict)`

- Reads `emotion_model` and `stress_model` from session state
- Applies `clean_text()` to user input
- Predicts emotion via `emotion_pipeline.predict()`
- Predicts stress level via TF-IDF transform + `stress_model.predict()`
- Shows confidence bars using `predict_proba` (or softmax of `decision_function` for SVM)
- Three quick-example buttons pre-fill the text area

---

## Data Requirements

Place these three files in a `data/` folder next to `app.py`:

| File | Required Columns | Description |
|---|---|---|
| `emotion_accuracy_training.csv` | `tweet`, `label` | Tweet text + emotion label string |
| `ugm_fess_labeled.csv` | `full_text`, `*label*` | Post text + numeric stress label (0/1/2) |
| `slang_indo.csv` | col 0 = slang, col 1 = formal | Indonesian slang normalization dictionary |

The stress label column is detected automatically — any column whose name contains `"label"` (case-insensitive) is used.

---

## Installation

**Python 3.8+ recommended.**

Install all dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn wordcloud PySastrawi
```

Or create a `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
wordcloud
PySastrawi
```

Then run:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
cd mindscan
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## App Pages

Navigate using the sidebar menu:

| Page | Purpose |
|---|---|
| 🏠 Home | Overview, dataset summary, usage guide |
| 📊 EDA | Explore data distributions, word clouds, slang analysis |
| ⚙️ Preprocessing | See and test the text cleaning pipeline |
| 🤖 Model Training | Configure and train ML models, view evaluation metrics |
| 🔮 Prediction | Detect emotion and stress from any text input |

> **Note:** You must train a model on the **Model Training** page before the **Prediction** page will work. Models are held in session state and reset when the browser tab is closed.

---

## ML Pipeline

```
Raw Text
   │
   ▼
clean_text()          ← lowercase, remove URLs/@/#/symbols,
   │                    collapse repeated chars, normalize slang
   ▼
TF-IDF Vectorizer     ← max_features configurable (1k–20k),
   │                    bigrams enabled (ngram_range=(1,2))
   ▼
Balancing Strategy    ← optional resampling on training split only
   │
   ▼
Classifier            ← Logistic Regression / Naive Bayes / Linear SVM
   │
   ▼
Predicted Label       ← stress: 0/1/2   emotion: string class
```

Both models share the same TF-IDF feature space for stress detection. The emotion model uses a full sklearn `Pipeline` so vectorization and classification happen in a single `.predict()` call.

---

## Text Preprocessing Pipeline

Each text goes through these steps in order inside `clean_text()`:

| Step | What it does | Example |
|---|---|---|
| Lowercase | Converts all characters to lowercase | `"Capek BANGET"` → `"capek banget"` |
| Remove URLs | Strips `http://`, `https://`, `www.` links | `"cek http://t.co/x"` → `"cek "` |
| Remove @/# | Strips mentions and hashtags | `"@teman #stress"` → `" "` |
| Remove non-alpha | Keeps only letters and spaces | `"capek!!!123"` → `"capek"` |
| Collapse repeats | Max 2 consecutive identical chars | `"capeeeeek"` → `"capee"` |
| Strip whitespace | Normalizes multiple spaces | `"capek  banget"` → `"capek banget"` |
| Normalize slang | Replaces slang with formal words | `"gw"` → `"saya"` |

---

## Balancing Strategies

Class imbalance is handled before fitting the stress classifier. Resampling is applied **only to the training split**, never to the test split.

| Strategy | Method | Best for |
|---|---|---|
| Random Oversampling | Duplicates minority class samples randomly | Quick fix, small datasets |
| SMOTE | Generates synthetic minority samples via interpolation | Medium datasets, smooth boundaries |
| Random Undersampling | Removes majority class samples randomly | Large datasets where data loss is acceptable |
| SMOTETomek | SMOTE + Tomek Links to clean boundary noise | Best quality, slower |
| Tanpa Balancing | No resampling | Already balanced data |
