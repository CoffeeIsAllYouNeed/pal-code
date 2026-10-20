"""Combining the ranked results of the vector index and the keyword index into one ranking."""
from __future__ import annotations

from localrag.embed.similarity import top_k


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Combine ranked lists by summing 1 / (k + rank) for each list an item appears in; scores themselves are ignored."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _) in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return top_k(fused.items(), len(fused))


def min_max(scored: list[tuple[str, float]]) -> dict[str, float]:
    """Rescale scores to the range 0 to 1, so scores from different retrievers can be added."""
    if not scored:
        return {}
    lo, hi = min(s for _, s in scored), max(s for _, s in scored)
    return {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in scored}


def weighted_fusion(vector_hits: list[tuple[str, float]], keyword_hits: list[tuple[str, float]], alpha: float = 0.5) -> list[tuple[str, float]]:
    """alpha times the rescaled vector score plus (1 - alpha) times the rescaled keyword score."""
    v, kw = min_max(vector_hits), min_max(keyword_hits)
    combined = {i: alpha * v.get(i, 0.0) + (1 - alpha) * kw.get(i, 0.0) for i in set(v) | set(kw)}
    return top_k(combined.items(), len(combined))
