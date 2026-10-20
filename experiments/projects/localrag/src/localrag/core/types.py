"""The data that moves through LocalRAG: documents, chunks, search hits and citations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """A source text that has been loaded: where it came from, its text and any metadata (title, language, ...)."""
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document, small enough to embed and to show as evidence. start and end are character offsets in the document."""
    id: str
    doc_id: str
    index: int
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(data["id"], data["doc_id"], int(data["index"]), data["text"], int(data["start"]), int(data["end"]),
                   dict(data.get("metadata", {})))


@dataclass
class SearchHit:
    """One retrieved chunk with its score, the retriever that found it ("vector", "keyword" or "hybrid") and its 1-based rank."""
    chunk: Chunk
    score: float
    retriever: str = "hybrid"
    rank: int = 0


@dataclass
class Citation:
    """Where an answer's evidence came from: the source, the character span in it, and a short snippet to show."""
    doc_id: str
    chunk_id: str
    source: str
    start: int
    end: int
    snippet: str


@dataclass
class QueryResult:
    """The outcome of a search: the hits, their citations, the context text to give a model, and how long it took."""
    query: str
    hits: list[SearchHit]
    citations: list[Citation]
    context: str
    took_ms: float = 0.0
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "took_ms": round(self.took_ms, 2), "from_cache": self.from_cache, "context": self.context,
                "hits": [{"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "score": round(h.score, 6), "rank": h.rank,
                          "retriever": h.retriever, "text": h.chunk.text} for h in self.hits],
                "citations": [asdict(c) for c in self.citations]}
