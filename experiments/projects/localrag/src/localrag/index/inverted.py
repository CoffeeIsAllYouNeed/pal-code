"""A keyword index scored with BM25: chunks that contain rare query words many times, and are not very long, score highest."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from localrag.embed.similarity import top_k


class InvertedIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = {}
        self._length: dict[str, int] = {}

    def add(self, item_id: str, tokens: list[str]) -> None:
        """Index the tokens of an item, replacing anything stored earlier under the same id."""
        self.remove(item_id)
        self._length[item_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            self._postings.setdefault(term, {})[item_id] = tf

    def remove(self, item_id: str) -> bool:
        if item_id not in self._length:
            return False
        del self._length[item_id]
        for term in [t for t, posting in self._postings.items() if item_id in posting]:
            del self._postings[term][item_id]
            if not self._postings[term]:
                del self._postings[term]
        return True

    def idf(self, term: str) -> float:
        """How informative a term is: high for rare terms, near zero for terms in almost every item."""
        n, df = len(self._length), len(self._postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """The k items with the highest BM25 score for the query terms, best first."""
        if not self._length:
            return []
        avg = sum(self._length.values()) / len(self._length)
        scores: dict[str, float] = {}
        for term in set(query_tokens):
            idf = self.idf(term)
            for item_id, tf in self._postings.get(term, {}).items():
                if allowed is not None and item_id not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._length[item_id] / avg)
                scores[item_id] = scores.get(item_id, 0.0) + idf * tf * (self.k1 + 1) / norm
        return top_k(scores.items(), k)

    def __len__(self) -> int:
        return len(self._length)

    def vocabulary_size(self) -> int:
        return len(self._postings)

    def to_dict(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b, "postings": self._postings, "length": self._length}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvertedIndex:
        index = cls(float(data["k1"]), float(data["b"]))
        index._postings = {t: {i: int(tf) for i, tf in p.items()} for t, p in data["postings"].items()}
        index._length = {i: int(n) for i, n in data["length"].items()}
        return index
