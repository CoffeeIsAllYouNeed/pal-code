"""Extractive summaries: pick the sentences that carry the most frequent content words, and keep them in their original order."""
from __future__ import annotations

from collections import Counter

from localrag.core.textutil import sentence_split, tokenize


def summarize(text: str, sentences: int = 2, max_chars: int = 280) -> str:
    """The `sentences` most central sentences of the text, in order, cut to max_chars. The first sentence gets a small bonus."""
    parts = sentence_split(" ".join(text.split()))
    if len(parts) <= sentences:
        return " ".join(parts)[:max_chars]
    freq = Counter(t for p in parts for t in tokenize(p, stem=True))
    scored = []
    for i, p in enumerate(parts):
        tokens = tokenize(p, stem=True)
        score = sum(freq[t] for t in tokens) / (len(tokens) or 1) + (1.0 if i == 0 else 0.0)
        scored.append((score, i))
    keep = sorted(i for _, i in sorted(scored, key=lambda s: (-s[0], s[1]))[:sentences])
    return " ".join(parts[i] for i in keep)[:max_chars]
