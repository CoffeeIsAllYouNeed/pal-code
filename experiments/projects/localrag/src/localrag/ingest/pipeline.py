"""Ingestion: bytes or text in, a Document and its Chunks out, with near-duplicate chunks dropped."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localrag.core.config import ChunkConfig
from localrag.core.errors import IngestError
from localrag.core.ids import make_doc_id
from localrag.core.log import get_logger
from localrag.core.types import Chunk, Document
from localrag.ingest.chunkers.registry import choose_chunker
from localrag.ingest.dedupe import NearDuplicateFilter
from localrag.ingest.loaders.registry import BY_EXTENSION, load_bytes

log = get_logger(__name__)
MAX_FILE_BYTES = 20_000_000


@dataclass
class IngestResult:
    """What one ingestion produced: the document, its kept chunks, and how many near-duplicates were dropped."""
    document: Document
    chunks: list[Chunk]
    duplicates_dropped: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestPipeline:
    def __init__(self, config: ChunkConfig, dedupe_threshold: float = 0.9):
        self.config = config
        self.dedupe_threshold = dedupe_threshold

    def ingest_bytes(self, raw: bytes, source: str, metadata: dict[str, Any] | None = None) -> IngestResult:
        """Load, chunk and de-duplicate one file's bytes. The chunks of a single file never duplicate each other."""
        if len(raw) > MAX_FILE_BYTES:
            raise IngestError(f"{source} is larger than {MAX_FILE_BYTES // 1_000_000} MB")
        loaded = load_bytes(raw, source)
        doc = Document(make_doc_id(source, loaded.text), source, loaded.text, {**loaded.metadata, **(metadata or {})})
        chunker = choose_chunker(self.config, source)
        seen, dropped = NearDuplicateFilter(self.dedupe_threshold), 0
        kept: list[Chunk] = []
        for chunk in chunker.chunk(doc):
            if seen.check_and_add(chunk.text):
                kept.append(Chunk(chunk.id.rsplit(":", 1)[0] + f":{len(kept):04d}", chunk.doc_id, len(kept), chunk.text, chunk.start,
                                  chunk.end, chunk.metadata))
            else:
                dropped += 1
        if not kept:
            raise IngestError(f"{source} produced no chunks")
        log.debug("ingested %s: %d chunks, %d duplicates dropped", source, len(kept), dropped)
        return IngestResult(doc, kept, dropped, {"chunker": chunker.name})

    def ingest_text(self, text: str, source: str = "text.txt", metadata: dict[str, Any] | None = None) -> IngestResult:
        """Ingest a string as if it were a file called source (its extension picks the loader and the chunker)."""
        return self.ingest_bytes(text.encode("utf-8"), source, metadata)

    def ingest_path(self, path: str | Path, recursive: bool = True) -> list[IngestResult]:
        """Ingest one file, or every supported file under a directory. Hidden files and unsupported types are skipped."""
        p = Path(path)
        if not p.exists():
            raise IngestError(f"path not found: {p}")
        files = [p] if p.is_file() else sorted(f for f in (p.rglob("*") if recursive else p.glob("*"))
                                              if f.is_file() and f.suffix.lower() in BY_EXTENSION
                                              and not any(part.startswith(".") for part in f.relative_to(p).parts))
        if not files:
            raise IngestError(f"no supported files under {p}")
        results = []
        for f in files:
            try:
                results.append(self.ingest_bytes(f.read_bytes(), f.relative_to(p).as_posix() if p.is_dir() else f.name,
                                                 {"path": str(f)}))
            except IngestError as e:
                if p.is_file():
                    raise
                log.warning("skipped %s: %s", f, e)
        if not results:
            raise IngestError(f"nothing under {p} could be ingested")
        return results
