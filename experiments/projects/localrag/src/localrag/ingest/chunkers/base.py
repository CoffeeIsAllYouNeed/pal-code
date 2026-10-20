"""What every chunker shares. A chunker only decides where to cut (as character spans); turning spans into Chunk objects is done here."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from localrag.core.config import ChunkConfig
from localrag.core.ids import make_chunk_id
from localrag.core.types import Chunk, Document


@dataclass(frozen=True)
class Span:
    """A stretch of a document, from start to end (exclusive), with what the chunker knows about it (heading, symbol, ...)."""
    start: int
    end: int
    meta: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    name = "base"

    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def split(self, text: str) -> list[Span]:
        """Return the spans to cut the text into, in order."""

    def merge_small(self, spans: list[Span]) -> list[Span]:
        """Join a span that is shorter than min_chunk_chars onto the one before it."""
        out: list[Span] = []
        for s in spans:
            if out and (s.end - s.start) < self.config.min_chunk_chars:
                prev = out[-1]
                out[-1] = Span(prev.start, max(prev.end, s.end), prev.meta)
            else:
                out.append(s)
        return out

    def chunk(self, doc: Document) -> list[Chunk]:
        """Cut a document into chunks with stable ids, trimmed text and the document's metadata plus each span's own."""
        chunks: list[Chunk] = []
        shared = {"source": doc.source, **{k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}}
        for span in self.merge_small(self.split(doc.text)):
            raw = doc.text[span.start:span.end]
            body = raw.strip()
            if not body:
                continue
            start = span.start + (len(raw) - len(raw.lstrip()))
            chunks.append(Chunk(make_chunk_id(doc.id, len(chunks)), doc.id, len(chunks), body, start, start + len(body), {**shared, **span.meta}))
        return chunks
