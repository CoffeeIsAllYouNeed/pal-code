"""Rerankers: re-score the candidates of the first search with cheap, precise signals, such as how many query words a chunk really contains."""
from __future__ import annotations

from abc import ABC, abstractmethod

from localrag.core.textutil import tokenize
from localrag.core.types import SearchHit


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query_tokens: list[str], hits: list[SearchHit]) -> list[SearchHit]:
        """Return the hits re-scored (scores between 0 and 1) and sorted best first."""


class NoopReranker(Reranker):
    name = "none"

    def rerank(self, query_tokens: list[str], hits: list[SearchHit]) -> list[SearchHit]:
        top = max((h.score for h in hits), default=0.0) or 1.0
        return [SearchHit(h.chunk, h.score / top, h.retriever, h.rank) for h in hits]


class TermOverlapReranker(Reranker):
    """Blend the first-stage score with the share of query words found in the chunk and a bonus for a query word pair appearing together.
    A chunk that contains none of the query words is probably only a near neighbour in vector space, so its score is cut to a quarter."""

    name = "term-overlap"

    def __init__(self, weight: float = 0.5, phrase_bonus: float = 0.15, no_overlap_penalty: float = 0.25):
        self.weight, self.phrase_bonus, self.no_overlap_penalty = weight, phrase_bonus, no_overlap_penalty

    def rerank(self, query_tokens: list[str], hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        top = max(h.score for h in hits) or 1.0
        wanted = set(query_tokens)
        pairs = {(a, b) for a, b in zip(query_tokens, query_tokens[1:], strict=False)}
        out = []
        for h in hits:
            tokens = tokenize(h.chunk.text, stem=True)
            overlap = len(wanted & set(tokens)) / len(wanted) if wanted else 0.0
            adjacent = {(a, b) for a, b in zip(tokens, tokens[1:], strict=False)}
            bonus = self.phrase_bonus * (len(pairs & adjacent) / len(pairs)) if pairs else 0.0
            score = min(1.0, (1 - self.weight) * (h.score / top) + self.weight * overlap + bonus)
            if overlap == 0.0:
                score *= self.no_overlap_penalty
            out.append(SearchHit(h.chunk, score, h.retriever, h.rank))
        out.sort(key=lambda h: (-h.score, h.chunk.id))
        return out
