"""Making a golden set from the index itself: for a sample of chunks, the query is the few most distinctive words of the chunk."""
from __future__ import annotations

import random

from localrag.core.textutil import tokenize
from localrag.engine import LocalRAG
from localrag.eval.golden import GoldenQuery


def distinctive_words(engine: LocalRAG, text: str, n: int = 4) -> list[str]:
    """The n words of the text with the highest idf in the keyword index (the rarest, so the most telling), in text order."""
    seen, scored = set(), []
    for position, word in enumerate(tokenize(text, stem=True)):
        if word not in seen and len(word) > 3:
            seen.add(word)
            scored.append((engine.index.keywords.idf(word), -position, word))
    best = sorted(scored, reverse=True)[:n]
    return [w for _, _, w in sorted(best, key=lambda s: -s[1], reverse=True)]


def generate_golden(engine: LocalRAG, n: int = 20, seed: int = 0) -> list[GoldenQuery]:
    """Up to n queries from randomly chosen chunks; each is expected to find the document its chunk came from."""
    rnd = random.Random(seed)
    chunks = engine.index.store.all_chunks()
    rnd.shuffle(chunks)
    items = []
    for chunk in chunks:
        words = distinctive_words(engine, chunk.text)
        if len(words) >= 3:
            items.append(GoldenQuery(" ".join(words), {str(chunk.metadata.get("source", chunk.doc_id))}))
        if len(items) >= n:
            break
    return items
