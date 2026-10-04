# 🧠 MindScan — NLP Stress Detection

A Streamlit web application that detects **emotions** and **stress levels** from Indonesian social media text using Natural Language Processing and Machine Learning.

## Authors
BINUS University Students:
1. Jonathan Raffael - 2802455275
2. ⁠Darren Star Limantoro - 2802461422
3. Steven Hosea - 2802453591
4. Albertus Adrian - 2802451876
5. Nicholas Driyadis Tjoe - 2802461321

---

## Table of Contents
- [Problem Statement (Masalah)](#problem-statement-masalah)
- [Datasets (Dataset)](#datasets-dataset)
- [Methodology (Metode)](#methodology-metode)
- [Results & Metrics (Hasil)](#results--metrics-hasil)
- [How to Run (Cara Menjalankan)](#how-to-run-cara-menjalankan)
- [Features](#features)
- [Project Structure & File Reference](#project-structure--file-reference)
- [Bulk Prediction & Clinical Analysis](#bulk-prediction--clinical-analysis)

---

## Problem Statement (Masalah)
Media sosial adalah salah satu wadah utama bagi masyarakat Indonesia untuk mengekspresikan perasaan dan keluh kesah. Namun, mengidentifikasi tanda-tanda stres dan emosi dari teks berbahasa Indonesia sangat menantang karena tingginya penggunaan bahasa gaul (slang), singkatan, dan struktur kalimat yang tidak baku. 

**MindScan** hadir untuk memecahkan masalah ini dengan menyediakan *pipeline* NLP otomatis yang mampu membersihkan teks informal Indonesia, mengekstraksi fitur, dan mengklasifikasikan teks ke dalam tingkat stres (Normal, Mild, High) serta kategori emosi (Happy, Sad, Anger, dll.) sebagai alat bantu deteksi dini kesehatan mental.

---

## Datasets (Dataset)
Aplikasi ini membutuhkan tiga dataset utama yang harus ditempatkan di dalam folder `data/`:

| File | Required Columns | Description |
|---|---|---|
| `emotion_accuracy_training.csv` | `tweet`, `label` | Dataset berisi teks tweet berbahasa Indonesia beserta label emosi (`happy`, `sad`, `anger`, `fear`, `love`, `surprise`, `neutral`). |
| `ugm_fess_labeled.csv` | `full_text`, `*label*` | Dataset postingan forum mahasiswa (UGM Fess) dengan label tingkat stres numerik (`0 = Normal`, `1 = Mild Stress`, `2 = High Stress`). |
| `slang_indo.csv` | `col 0` (slang), `col 1` (formal)| Kamus pemetaan kata gaul/informal bahasa Indonesia ke bentuk baku untuk proses normalisasi. |

---

## Methodology (Metode)

### 1. Text Preprocessing Pipeline
Setiap teks mentah akan diproses melalui tahapan berurutan (`clean_text`):
- **Case Folding:** Mengubah semua huruf menjadi *lowercase*.
- **Noise Removal:** Menghapus URL, mention (`@`), hashtag (`#`), dan karakter non-alfabet.
- **Character Normalization:** Memperbaiki huruf yang diketik berulang (contoh: `"capeeeeek"` → `"capee"`).
- **Slang Normalization:** Mengubah kata gaul menjadi kata baku menggunakan `slang_indo.csv` (contoh: `"gw"` → `"saya"`).
- **Stemming (Opsional):** Menggunakan library `PySastrawi` untuk mengembalikan kata ke bentuk dasarnya.

### 2. Feature Extraction (TF-IDF)
Teks yang telah bersih diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**. Parameter seperti `max_features` (1k–20k) dan penggunaan bigram (`ngram_range=(1,2)`) dapat dikonfigurasi langsung melalui UI.

### 3. Balancing Strategies
Untuk menangani ketidakseimbangan kelas pada dataset stres, teknik *resampling* diterapkan **hanya pada data training**:
- Random Oversampling / Undersampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTETomek (Kombinasi oversampling dan pembersihan *noise* data)

### 4. Machine Learning Models
Data yang telah diseimbangkan dilatih menggunakan salah satu algoritma klasifikasi tradisional berikut:
- **Logistic Regression**
- **Naive Bayes**
- **Linear SVM (Support Vector Machine)**

---

## Results & Metrics (Hasil)
Karena MindScan dirancang secara interaktif, hasil dan metrik model dihasilkan secara *real-time* berdasarkan konfigurasi (pilihan model, proporsi *test split*, dan strategi *balancing*) yang diatur oleh pengguna di halaman **Model Training**. 

Metrik yang dihitung dan divisualisasikan meliputi:
- **Global Metrics:** Accuracy, Precision, Recall, F1-Score.
- **Classification Report:** Metrik detail untuk setiap kelas (Stres 0/1/2 dan 7 kelas Emosi).
- **Confusion Matrix:** Heatmap yang menunjukkan prediksi benar vs salah.
- **Keyword Importance:** Visualisasi kata-kata paling berpengaruh untuk setiap kelas (khusus Logistic Regression dan SVM).

---

## How to Run (Cara Menjalankan)

**Prerequisites:** Python 3.8+ direkomendasikan.

**1. Install Dependencies**
Gunakan `pip` untuk menginstal seluruh library yang dibutuhkan:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn wordcloud PySastrawi
```

**2. Siapkan Dataset**
Pastikan folder `data/` telah berisi file `emotion_accuracy_training.csv`, `ugm_fess_labeled.csv`, dan `slang_indo.csv` sejajar dengan file `app.py`.

**3. Jalankan Aplikasi Streamlit**
Buka terminal/command prompt, navigasikan ke direktori proyek, lalu jalankan:
```bash
streamlit run app.py
```
Aplikasi akan terbuka secara otomatis di browser pada `http://localhost:8501`.

---

## Features
- 📊 **EDA (Exploratory Data Analysis)** — Distribusi data, Word Clouds, dan analisis kata *slang*.
- ⚙️ **Interactive Preprocessing** — Visualisasi *live tokenization*, perbandingan teks sebelum/sesudah *cleaning*, dan uji coba *preprocessing* manual.
- 🤖 **Model Training** — Konfigurasi hyperparameter, pemilihan metode *balancing*, dan pelatihan model *real-time* dengan metrik evaluasi lengkap.
- 🔮 **Prediction** — Deteksi teks tunggal atau analisis CSV (Bulk/Clinical) secara langsung dengan *confidence scores*.
- 🌙 **Dark UI** — Desain antarmuka *bento-style* yang modern dengan *dark theme* kustom.

---

## Project Structure & File Reference

```text
mindscan/
│
├── app.py                  # Entry point — page routing, sidebar, data bootstrap
├── styles.py               # Custom CSS styles dan matplotlib dark theme
├── data_loader.py          # Script untuk memuat dataset, caching, dan mapping label
├── utils.py                # Fungsi NLP murni (cleaning, stemming, slang normalization)
├── models.py               # ML model factory & resampling logic (SMOTE, dll)
│
├── views/                  # UI Pages (dipanggil oleh app.py)
│   ├── home.py             # 🏠 Home page
│   ├── eda.py              # 📊 EDA page
│   ├── preprocessing.py    # ⚙️ Preprocessing page
│   ├── training.py         # 🤖 Model Training page
│   └── prediction.py       # 🔮 Prediction page
│
└── data/                   # Folder Dataset
    ├── emotion_accuracy_training.csv
    ├── ugm_fess_labeled.csv
    └── slang_indo.csv
```

---

## Bulk Prediction & Clinical Analysis

Pada tab **Analisis CSV Bulk (Clinical)** di halaman Prediction, pengguna dapat mengunggah file CSV berisi riwayat postingan satu pengguna untuk dianalisis secara keseluruhan.

**Confidence-Weighted Majority Voting:**
Sistem tidak hanya mengambil label yang paling sering muncul (*majority vote*), melainkan menjumlahkan *confidence score* dari prediksi setiap postingan. Hal ini membuat kesimpulan klinis lebih stabil terhadap postingan yang ambigu atau ber-noise.

**Output Klinis:**
- Tingkat stres & emosi dominan secara keseluruhan.
- Interpretasi klinis dan rekomendasi (misal: rujukan ke psikolog/psikiater jika mayoritas High Stress).
- *Timeline Chart* fluktuasi skor per postingan.
- File CSV yang dapat diunduh berisi detail prediksi tiap baris dan ringkasan klinis.
