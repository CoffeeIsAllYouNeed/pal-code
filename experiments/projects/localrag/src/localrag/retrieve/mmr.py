"""Maximal marginal relevance: pick results that are relevant to the query but not near-copies of each other."""
from __future__ import annotations

from localrag.core.types import SearchHit
from localrag.embed.similarity import dot


def mmr_select(hits: list[SearchHit], vectors: dict[str, list[float]], k: int, lambda_: float = 0.7) -> list[SearchHit]:
    """Greedily choose k hits. Each step takes the hit with the best (lambda * relevance - (1 - lambda) * similarity to those already chosen)."""
    remaining: list[SearchHit] = list(hits)
    chosen: list[SearchHit] = []
    while remaining and len(chosen) < k:
        def value(h: SearchHit) -> float:
            redundancy = max((dot(vectors[h.chunk.id], vectors[c.chunk.id]) for c in chosen), default=0.0)
            return lambda_ * h.score - (1 - lambda_) * redundancy
        best = max(remaining, key=lambda h: (value(h), h.score))
        chosen.append(best)
        remaining.remove(best)
    return chosen
