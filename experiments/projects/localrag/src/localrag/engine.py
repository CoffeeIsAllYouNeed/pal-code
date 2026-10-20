"""The LocalRAG engine: one object that ingests documents, answers queries, searches the document tree, and saves and loads its state."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from localrag.core.config import Config, config_from_dict
from localrag.core.errors import IndexingError, RetrievalError
from localrag.core.types import Chunk, Document, QueryResult
from localrag.embed.registry import build_embedder
from localrag.index.hybrid import HybridIndex
from localrag.index.persistence import load_snapshot, save_snapshot
from localrag.ingest.pipeline import IngestPipeline, IngestResult
from localrag.retrieve.pipeline import RetrievalPipeline
from localrag.tree.builder import build_tree
from localrag.tree.render import render_outline, section_text
from localrag.tree.search import TreeIndex


class LocalRAG:
    def __init__(self, config: Config | None = None):
        self.config = (config or Config()).validate()
        self.embedder = build_embedder(self.config.embed)
        self.index = HybridIndex(self.config.index, self.embedder)
        self.ingestor = IngestPipeline(self.config.chunk)
        self.retrieval = RetrievalPipeline(self.config.retrieve, self.index)
        self.trees = TreeIndex()
        self._lock = threading.RLock()

    # -- ingestion ---------------------------------------------------------------------------------------------------------------
    def _register(self, result: IngestResult) -> dict[str, Any]:
        with self._lock:
            self.index.add_document(result.document, result.chunks)
            self.trees.add(result.document.id, result.document.source,
                           build_tree(result.document.id, str(result.document.metadata.get("title", result.document.source)),
                                      result.document.text, self.config.tree), result.document.text)
            self.retrieval.invalidate()
        return {"doc_id": result.document.id, "source": result.document.source, "chunks": len(result.chunks),
                "duplicates_dropped": result.duplicates_dropped, "chunker": result.metadata.get("chunker"),
                "characters": len(result.document.text)}

    def ingest_text(self, text: str, source: str = "text.txt", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Index a piece of text as a document called source (its extension picks how it is read and cut)."""
        return self._register(self.ingestor.ingest_text(text, source, metadata))

    def ingest_path(self, path: str | Path, recursive: bool = True) -> list[dict[str, Any]]:
        """Index one file or every supported file under a folder."""
        return [self._register(r) for r in self.ingestor.ingest_path(path, recursive)]

    # -- queries -----------------------------------------------------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> QueryResult:
        """The best chunks for a query, with citations and a context text."""
        with self._lock:
            return self.retrieval.search(query, top_k, filters)

    def tree_search(self, query: str, top_n: int = 5) -> list[dict[str, Any]]:
        """The best-matching sections found by walking document trees, without using vectors."""
        with self._lock:
            matches = self.trees.search(query, self.config.tree.tree_beam, top_n)
            return [{"doc_id": m.doc_id, "source": m.source, "section": m.node.title, "path": m.path, "score": round(m.score, 3),
                     "summary": m.node.summary, "text": section_text(self.index.store.get_document(m.doc_id).text, m.node, 800)}
                    for m in matches]

    def outline(self, name: str) -> str:
        """The heading outline of a document, given its id or source name."""
        doc = self.get_document(name)
        root = self.trees.get(doc.id)
        if root is None:
            raise RetrievalError(f"no tree for {doc.source}")
        return render_outline(root, self.config.tree.tree_max_depth)

    # -- access ------------------------------------------------------------------------------------------------------------------
    def get_document(self, name: str) -> Document:
        """A document by id or by source name."""
        doc = self.index.store.find_document(name)
        if doc is None:
            raise IndexingError(f"no document {name!r} (give its id or its exact source name)")
        return doc

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.index.store.get_chunk(chunk_id)

    def list_documents(self) -> list[dict[str, Any]]:
        return [{"doc_id": d.id, "source": d.source, "characters": len(d.text), "chunks": len(self.index.store.chunks_of(d.id)),
                 "title": d.metadata.get("title", "")} for d in self.index.store.documents()]

    def delete_document(self, name: str) -> int:
        """Remove a document everywhere; returns how many chunks were removed."""
        with self._lock:
            doc = self.get_document(name)
            self.trees.remove(doc.id)
            removed = self.index.remove_document(doc.id)
            self.retrieval.invalidate()
            return removed

    def stats(self) -> dict[str, Any]:
        return {**self.index.store.stats(), "vocabulary": self.index.keywords.vocabulary_size(), "vector_index": self.index.vectors.kind,
                "embedding_dim": self.embedder.dim, "trees": len(self.trees), "query_cache": self.retrieval.cache.stats(),
                "embedding_cache": self.embedder.stats() if hasattr(self.embedder, "stats") else {}}

    # -- saving ------------------------------------------------------------------------------------------------------------------
    def snapshot_path(self, path: str | Path | None = None) -> Path:
        return Path(path) if path else Path(self.config.data_dir) / "snapshot.json"

    def save(self, path: str | Path | None = None) -> int:
        """Write the whole state to a snapshot file; returns its size in bytes."""
        with self._lock:
            return save_snapshot(self.snapshot_path(path), {"config": self.config.to_dict(), "store": self.index.store.to_dict()})

    @classmethod
    def load(cls, path: str | Path) -> LocalRAG:
        """Rebuild an engine from a snapshot: the text is restored and the indexes and trees are recomputed from it."""
        payload = load_snapshot(path)
        engine = cls(config_from_dict(payload["config"]))
        from localrag.index.store import ChunkStore
        engine.index.store = ChunkStore.from_dict(payload["store"])
        engine.index.rebuild()
        for doc in engine.index.store.documents():
            engine.trees.add(doc.id, doc.source, build_tree(doc.id, str(doc.metadata.get("title", doc.source)), doc.text, engine.config.tree), doc.text)
        engine.retrieval.invalidate()
        return engine
