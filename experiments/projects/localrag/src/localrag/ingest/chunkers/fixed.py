"""Fixed-size windows that overlap, cut at a space or a newline near the end of each window."""
from __future__ import annotations

from localrag.ingest.chunkers.base import Chunker, Span


class FixedChunker(Chunker):
    name = "fixed"

    def split(self, text: str) -> list[Span]:
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        spans: list[Span] = []
        pos, n = 0, len(text)
        while pos < n:
            end = min(n, pos + size)
            if end < n:
                floor = pos + int(size * 0.8)
                cut = max(text.rfind(" ", floor, end), text.rfind("\n", floor, end))
                if cut > pos:
                    end = cut
            spans.append(Span(pos, end))
            if end >= n:
                break
            pos = max(end - overlap, pos + 1)
        return spans
