"""
utils.py — Text cleaning, slang normalization, stopword removal, stemming.
Sastrawi objects are initialized ONCE at module level.
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Sastrawi singletons ───────────────────────────────────────────────────────
stemmer          = StemmerFactory().create_stemmer()
_stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# ── Words that must survive stopword removal ──────────────────────────────────
# Negations, intensifiers, and emotional keywords are critical stress signals.
KEEP_WORDS = {
    # negations
    "tidak","bukan","jangan","belum","tak","ga","gak","nggak",
    # intensifiers
    "sangat","banget","sekali","terlalu","lebih","paling",
    # stress / emotion vocabulary
    "capek","lelah","stress","stres","sedih","marah","takut",
    "cemas","khawatir","galau","bingung","panik","putus",
    "susah","sulit","berat","buruk","jelek","hancur","gagal",
    "sakit","nyeri","pusing","mual","lemas","lemah",
    "sendiri","sepi","kesepian","ditinggal","diabaikan",
    "nangis","menangis","menjerit","teriak",
    "mati","bunuh","hilang","pergi","kabur","lari",
    "harap","semoga","ingin","mau","pengen","butuh",
    "baik","senang","suka","cinta","bahagia","gembira",
}

# ── Pre-compiled regex patterns (compiled once, reused) ───────────────────────
_RE_URL      = re.compile(r"http\S+|www\S+")
_RE_MENTION  = re.compile(r"@\w+|#\w+")
_RE_NONALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_REPEAT   = re.compile(r"(.)\1{2,}")
_RE_SPACE    = re.compile(r"\s+")


def remove_repeated_char(text: str) -> str:
    return _RE_REPEAT.sub(r"\1\1", text)


def normalize_slang(text: str, slang_dict: dict) -> str:
    return " ".join(slang_dict.get(w, w) for w in text.split())


def remove_stopwords(text: str) -> str:
    """Sastrawi stopword removal, preserving KEEP_WORDS."""
    kept = []
    for word in text.split():
        if word in KEEP_WORDS:
            kept.append(word)
        else:
            result = _stopword_remover.remove(word).strip()
            if result:
                kept.append(result)
    return " ".join(kept)


def clean_text(text: str, slang_dict: dict) -> str:
    """Full preprocessing pipeline."""
    t = str(text).lower()
    t = _RE_URL.sub("", t)
    t = _RE_MENTION.sub("", t)
    t = _RE_NONALPHA.sub(" ", t)
    t = remove_repeated_char(t)
    t = _RE_SPACE.sub(" ", t).strip()
    t = normalize_slang(t, slang_dict)
    t = remove_stopwords(t)
    t = stemmer.stem(t)
    return t


def detect_slang_words(text: str, slang_words: set) -> list:
    return [w for w in str(text).lower().split() if w in slang_words]
