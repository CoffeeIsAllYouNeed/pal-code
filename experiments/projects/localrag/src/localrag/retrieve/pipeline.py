"""The retrieval pipeline: process the query, search both indexes, rerank, diversify, then build citations and the context text."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from localrag.core.config import RetrieveConfig
from localrag.core.errors import RetrievalError
from localrag.core.types import QueryResult
from localrag.index.hybrid import HybridIndex
from localrag.retrieve.cache import QueryCache
from localrag.retrieve.citations import build_citations
from localrag.retrieve.context import build_context
from localrag.retrieve.mmr import mmr_select
from localrag.retrieve.query import process_query
from localrag.retrieve.rerank import NoopReranker, Reranker, TermOverlapReranker


class RetrievalPipeline:
    def __init__(self, config: RetrieveConfig, index: HybridIndex, reranker: Reranker | None = None):
        self.config, self.index = config, index
        self.reranker = reranker or (TermOverlapReranker() if config.use_reranker else NoopReranker())
        self.cache = QueryCache(config.query_cache_size)

    def invalidate(self) -> None:
        """Forget cached results; call after the index changes."""
        self.cache.clear()

    def search(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> QueryResult:
        """Answer a query with the best chunks, their citations and a ready-to-use context text."""
        started = time.perf_counter()
        k = top_k or self.config.top_k
        if k < 1:
            raise RetrievalError("top_k must be at least 1")
        q = process_query(query, filters)
        key = QueryCache.key(q.text, k, q.filters)
        cached = self.cache.get(key)
        if cached is not None:
            return replace(cached, from_cache=True, took_ms=(time.perf_counter() - started) * 1000)
        candidates = self.index.search(q.text, max(self.config.candidate_k, k), q.filters or None)
        ranked = self.reranker.rerank(q.tokens, candidates)
        vectors = {h.chunk.id: self.index.vectors.vector(h.chunk.id) for h in ranked}
        chosen = mmr_select([h for h in ranked if h.score >= self.config.min_score], vectors, k, self.config.mmr_lambda)
        hits = [replace(h, rank=i) for i, h in enumerate(chosen, start=1)]
        result = QueryResult(query, hits, build_citations(hits, q.text), build_context(hits, self.config.context_tokens),
                             (time.perf_counter() - started) * 1000)
        self.cache.put(key, result)
        return result
