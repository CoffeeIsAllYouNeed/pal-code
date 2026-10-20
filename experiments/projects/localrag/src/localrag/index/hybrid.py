"""The hybrid index: a vector index and a keyword index over the same chunks, searched together and fused into one ranking."""
from __future__ import annotations

from localrag.core.config import IndexConfig
from localrag.core.errors import IndexingError
from localrag.core.textutil import tokenize
from localrag.core.types import Chunk, Document, SearchHit
from localrag.embed.base import Embedder
from localrag.index.filters import matches
from localrag.index.fusion import reciprocal_rank_fusion, weighted_fusion
from localrag.index.inverted import InvertedIndex
from localrag.index.ivf import IVFIndex
from localrag.index.store import ChunkStore
from localrag.index.vector import FlatVectorIndex


def index_text(chunk: Chunk) -> str:
    """The text that represents a chunk in the indexes: its body plus its heading or symbol name, which often say what it is about."""
    extra = " ".join(str(chunk.metadata.get(k, "")) for k in ("heading", "symbol", "title"))
    return f"{extra}\n{chunk.text}" if extra.strip() else chunk.text


class HybridIndex:
    def __init__(self, config: IndexConfig, embedder: Embedder, store: ChunkStore | None = None):
        self.config, self.embedder = config, embedder
        self.store = store or ChunkStore()
        self.vectors: FlatVectorIndex = (IVFIndex(embedder.dim, config.ivf_lists, config.ivf_probes) if config.vector_index == "ivf"
                                         else FlatVectorIndex(embedder.dim))
        self.keywords = InvertedIndex(config.bm25_k1, config.bm25_b)

    def add_document(self, doc: Document, chunks: list[Chunk]) -> int:
        """Index a document's chunks in both indexes, replacing the document if it was indexed before. Returns the chunks added."""
        for old in self.store.add_document(doc, chunks):
            self.vectors.remove(old)
            self.keywords.remove(old)
        for chunk, vec in zip(chunks, self.embedder.embed([index_text(c) for c in chunks]), strict=True):
            self.vectors.add(chunk.id, vec)
            self.keywords.add(chunk.id, tokenize(index_text(chunk), stem=True))
        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Remove a document from the store and both indexes; returns how many chunks went."""
        removed = self.store.remove_document(doc_id)
        for chunk_id in removed:
            self.vectors.remove(chunk_id)
            self.keywords.remove(chunk_id)
        return len(removed)

    def allowed_ids(self, flt: dict | None) -> set[str] | None:
        """The ids of chunks whose metadata passes the filter; None means no filter."""
        if not flt:
            return None
        return {c.id for c in self.store.all_chunks() if matches(c.metadata, flt)}

    def search_vector(self, text: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        return self.vectors.search(self.embedder.embed_one(text), k, allowed)

    def search_keyword(self, text: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        return self.keywords.search(tokenize(text, stem=True), k, allowed)

    def search(self, text: str, k: int, flt: dict | None = None) -> list[SearchHit]:
        """The k best chunks for a query, found by both indexes and fused. A chunk found by only one index still counts."""
        if not text.strip():
            raise IndexingError("cannot search for an empty query")
        allowed = self.allowed_ids(flt)
        by_vector, by_keyword = self.search_vector(text, k * 2, allowed), self.search_keyword(text, k * 2, allowed)
        if self.config.fusion_method == "weighted":
            fused = weighted_fusion(by_vector, by_keyword, self.config.hybrid_alpha)
        else:
            fused = reciprocal_rank_fusion([by_vector, by_keyword], self.config.rrf_k)
        vec_ids, kw_ids = {i for i, _ in by_vector}, {i for i, _ in by_keyword}
        hits = []
        for rank, (chunk_id, score) in enumerate(fused[:k], start=1):
            source = "hybrid" if chunk_id in vec_ids and chunk_id in kw_ids else ("vector" if chunk_id in vec_ids else "keyword")
            hits.append(SearchHit(self.store.get_chunk(chunk_id), score, source, rank))
        return hits

    def rebuild(self) -> None:
        """Recompute both indexes from the store (used after loading a snapshot, which stores text but not vectors)."""
        chunks = self.store.all_chunks()
        self.vectors = type(self.vectors)(self.embedder.dim, *(([self.config.ivf_lists, self.config.ivf_probes]) if self.config.vector_index == "ivf" else []))
        self.keywords = InvertedIndex(self.config.bm25_k1, self.config.bm25_b)
        for chunk, vec in zip(chunks, self.embedder.embed([index_text(c) for c in chunks]), strict=True):
            self.vectors.add(chunk.id, vec)
            self.keywords.add(chunk.id, tokenize(index_text(chunk), stem=True))
