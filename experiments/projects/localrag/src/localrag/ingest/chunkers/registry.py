"""Picks a chunker from the configured strategy; "auto" chooses by file type."""
from __future__ import annotations

from pathlib import PurePosixPath

from localrag.core.config import ChunkConfig
from localrag.core.errors import IngestError
from localrag.ingest.chunkers.base import Chunker
from localrag.ingest.chunkers.code import CodeChunker
from localrag.ingest.chunkers.fixed import FixedChunker
from localrag.ingest.chunkers.markdown import MarkdownChunker
from localrag.ingest.chunkers.sentence import SentenceChunker
from localrag.ingest.loaders.code import LANGUAGES

CHUNKERS: dict[str, type[Chunker]] = {"fixed": FixedChunker, "sentence": SentenceChunker, "markdown": MarkdownChunker, "code": CodeChunker}


def strategy_for(name: str) -> str:
    """The chunking strategy "auto" picks for a file name: headings for markdown and html, syntax for code, sentences otherwise."""
    ext = PurePosixPath(name).suffix.lower()
    if ext in (".md", ".markdown", ".html", ".htm", ".xhtml"):
        return "markdown"
    return "code" if ext in LANGUAGES else "sentence"


def choose_chunker(config: ChunkConfig, name: str = "") -> Chunker:
    """The chunker for a file: the configured strategy, or the one that suits the file when the strategy is auto."""
    strategy = strategy_for(name) if config.chunk_strategy == "auto" else config.chunk_strategy
    if strategy not in CHUNKERS:
        raise IngestError(f"unknown chunking strategy {strategy!r}")
    return CHUNKERS[strategy](config)
