"""Groups whole sentences until a chunk is about chunk_size characters, repeating the last sentence at the start of the next chunk."""
from __future__ import annotations

from localrag.core.textutil import sentence_spans
from localrag.ingest.chunkers.base import Chunker, Span
from localrag.ingest.chunkers.fixed import FixedChunker


class SentenceChunker(Chunker):
    name = "sentence"

    def split(self, text: str) -> list[Span]:
        size = self.config.chunk_size
        spans: list[Span] = []
        group: list[tuple[int, int]] = []

        def flush() -> None:
            if group:
                spans.append(Span(group[0][0], group[-1][1]))

        for start, end in sentence_spans(text):
            if end - start > size * 2:                       # one enormous "sentence": fall back to windows
                flush()
                group.clear()
                spans.extend(Span(start + s.start, start + s.end) for s in FixedChunker(self.config).split(text[start:end]))
                continue
            if group and end - group[0][0] > size:
                flush()
                keep = group[-1] if self.config.chunk_overlap and (group[-1][1] - group[-1][0]) <= self.config.chunk_overlap * 2 else None
                group.clear()
                if keep:
                    group.append(keep)
            group.append((start, end))
        flush()
        return spans
