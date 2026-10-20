"""The chunk store: every document and chunk that has been ingested, kept so results can show their text and sources."""
from __future__ import annotations

from typing import Any

from localrag.core.errors import IndexingError
from localrag.core.types import Chunk, Document


class ChunkStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[str]] = {}

    def add_document(self, doc: Document, chunks: list[Chunk]) -> list[str]:
        """Store a document and its chunks, replacing an earlier version of the same document. Returns the ids that were replaced."""
        replaced = self.remove_document(doc.id)
        self._docs[doc.id] = doc
        self._by_doc[doc.id] = [c.id for c in chunks]
        self._chunks.update({c.id: c for c in chunks})
        return replaced

    def remove_document(self, doc_id: str) -> list[str]:
        """Forget a document and its chunks; returns the chunk ids removed (empty if the document was unknown)."""
        ids = self._by_doc.pop(doc_id, [])
        for chunk_id in ids:
            self._chunks.pop(chunk_id, None)
        self._docs.pop(doc_id, None)
        return ids

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise IndexingError(f"unknown chunk {chunk_id}")
        return self._chunks[chunk_id]

    def get_document(self, doc_id: str) -> Document:
        if doc_id not in self._docs:
            raise IndexingError(f"unknown document {doc_id}")
        return self._docs[doc_id]

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return [self._chunks[i] for i in self._by_doc.get(doc_id, [])]

    def documents(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.source)

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[i] for ids in self._by_doc.values() for i in ids]

    def find_document(self, name: str) -> Document | None:
        """A document by id, or by its source name when that is unambiguous."""
        if name in self._docs:
            return self._docs[name]
        named = [d for d in self._docs.values() if d.source == name]
        return named[0] if len(named) == 1 else None

    def stats(self) -> dict[str, int]:
        return {"documents": len(self._docs), "chunks": len(self._chunks), "characters": sum(len(d.text) for d in self._docs.values())}

    def to_dict(self) -> dict[str, Any]:
        return {"documents": [{"id": d.id, "source": d.source, "text": d.text, "metadata": d.metadata} for d in self._docs.values()],
                "chunks": [c.to_dict() for c in self.all_chunks()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkStore:
        store = cls()
        docs = {d["id"]: Document(d["id"], d["source"], d["text"], dict(d.get("metadata", {}))) for d in data["documents"]}
        chunks: dict[str, list[Chunk]] = {i: [] for i in docs}
        for raw in data["chunks"]:
            chunk = Chunk.from_dict(raw)
            chunks.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, doc in docs.items():
            store.add_document(doc, sorted(chunks[doc_id], key=lambda c: c.index))
        return store
