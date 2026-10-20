"""Small text helpers shared by ingestion, indexing and retrieval."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its of on or she that the their them there they this to was were "
    "will with you your we our not no can could should would may might do does did so if then than too very".split())
_WORD = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Unicode-normalise, drop control characters and collapse runs of blanks, keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def stem_word(word: str) -> str:
    """A light suffix stripper: purrs and purring both become purr. It is deliberately simple and only touches longer words."""
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not (suffix == "s" and word.endswith("ss")):
            stem = word[: -len(suffix)] + keep
            return stem[:-1] if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls" else stem
    return word


def tokenize(text: str, drop_stopwords: bool = True, min_len: int = 2, stem: bool = False) -> list[str]:
    """Lower-cased word tokens; stopwords and very short tokens are dropped unless asked otherwise, and words can be stemmed."""
    words = [w.lower() for w in _WORD.findall(text)]
    kept = [w for w in words if len(w) >= min_len and not (drop_stopwords and w in STOPWORDS)]
    return [stem_word(w) for w in kept] if stem else kept


def char_ngrams(text: str, n: int) -> list[str]:
    """Overlapping character n-grams of the lower-cased text, spaces included."""
    t = " " + " ".join(text.lower().split()) + " "
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences at ., ! or ? followed by a capital letter, a digit or an opening quote."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def estimate_tokens(text: str) -> int:
    """A cheap token estimate: one token per four characters, at least one for non-empty text."""
    return max(1, len(text) // 4) if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text to roughly max_tokens tokens, at a word boundary, adding an ellipsis when something was removed."""
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets (start, end) of each sentence in text, so chunks can point back into the original."""
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        if text[pos:m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans
