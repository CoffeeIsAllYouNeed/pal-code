"""Vector arithmetic in plain Python: enough for LocalRAG's small and medium collections without any native dependency."""
from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """The dot product of two vectors of the same length."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(v: Sequence[float]) -> float:
    """The Euclidean length of a vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Sequence[float]) -> list[float]:
    """The vector scaled to length one; the zero vector is returned unchanged."""
    n = norm(v)
    return [x / n for x in v] if n else list(v)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """The cosine of the angle between two vectors: 1 for the same direction, 0 for unrelated, -1 for opposite."""
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else 0.0


def centroid(vectors: Iterable[Sequence[float]], dim: int) -> list[float]:
    """The average of some vectors (zeros when there are none)."""
    total, count = [0.0] * dim, 0
    for v in vectors:
        for i, x in enumerate(v):
            total[i] += x
        count += 1
    return [x / count for x in total] if count else total


def top_k(scored: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """The k highest-scoring (id, score) pairs, best first; ties are broken by id so results are repeatable."""
    return heapq.nsmallest(k, scored, key=lambda p: (-p[1], p[0]))
