"""Ranking metrics. A ranking is a list of ids, best first; the relevant ids are a set. All gains are binary (relevant or not)."""
from __future__ import annotations

import math


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """The share of relevant items found in the first k results."""
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """The share of the first k results that are relevant."""
    top = ranked[:k]
    return sum(1 for r in top if r in relevant) / len(top) if top else 0.0


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    """1 divided by the rank of the first relevant result; 0 if none was found."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision(relevant: set[str], ranked: list[str]) -> float:
    """The mean of the precision values at each position where a relevant item appears."""
    hits, total = 0, 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant) if relevant else 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Normalised discounted cumulative gain: rewards relevant results more the higher they are ranked."""
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
