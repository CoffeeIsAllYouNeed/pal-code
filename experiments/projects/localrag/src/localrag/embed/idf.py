"""Inverse document frequency: how rare a word is across the collection. Rare words say more about what a chunk is about."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

from localrag.core.textutil import tokenize


class IdfTable:
    def __init__(self) -> None:
        self.docs = 0
        self.df: Counter[str] = Counter()

    def fit(self, texts: Iterable[str]) -> IdfTable:
        """Count, for each word, how many of the texts contain it."""
        for text in texts:
            self.docs += 1
            self.df.update(set(tokenize(text)))
        return self

    def weight(self, token: str) -> float:
        """Smoothed idf; words never seen get the highest weight, and every weight is at least 1."""
        return 1.0 + math.log((1 + self.docs) / (1 + self.df.get(token, 0)))

    def to_dict(self) -> dict[str, Any]:
        return {"docs": self.docs, "df": dict(self.df)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdfTable:
        table = cls()
        table.docs, table.df = int(data["docs"]), Counter({k: int(v) for k, v in data["df"].items()})
        return table
