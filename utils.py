"""
utils.py — Text cleaning, slang normalization, stemming, and NLP helpers.
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# STEMMER (initialized once)
# ==========================================
_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()


# ==========================================
# TEXT CLEANING PIPELINE
# ==========================================
def remove_repeated_char(text: str) -> str:
    """Collapse 3+ repeated characters to 2 (e.g. 'capeeek' → 'capee')."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_slang(text: str, slang_dict: dict) -> str:
    """Replace slang words with their formal equivalents."""
    words = text.split()
    return " ".join(slang_dict.get(word, word) for word in words)


def clean_text(text: str, slang_dict: dict) -> str:
    """
    Full cleaning pipeline:
    lowercase → strip URLs → strip @/# → keep alpha only →
    collapse repeated chars → strip whitespace → normalize slang.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = remove_repeated_char(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_slang(text, slang_dict)
    return text


# ==========================================
# SLANG DETECTION
# ==========================================
def detect_slang_words(text: str, slang_words: set) -> list:
    """Return list of slang words found in text."""
    words = str(text).lower().split()
    return [word for word in words if word in slang_words]
