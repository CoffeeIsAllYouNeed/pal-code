"""Cuts markdown at its headings, so a chunk never mixes two sections, and notes the heading path of each chunk."""
from __future__ import annotations

import re

from localrag.ingest.chunkers.base import Chunker, Span
from localrag.ingest.chunkers.sentence import SentenceChunker

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


class MarkdownChunker(Chunker):
    name = "markdown"

    def sections(self, text: str) -> list[tuple[int, int, str, int]]:
        """(start, end, heading path, level) for each section, in order; text before the first heading is one section."""
        marks = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text)]
        if not marks:
            return [(0, len(text), "", 0)]
        out: list[tuple[int, int, str, int]] = []
        path: list[tuple[int, str]] = []
        if marks[0][0] > 0 and text[:marks[0][0]].strip():
            out.append((0, marks[0][0], "", 0))
        for i, (pos, level, title) in enumerate(marks):
            path = [p for p in path if p[0] < level] + [(level, title)]
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            out.append((pos, end, " > ".join(t for _, t in path), level))
        return out

    def split(self, text: str) -> list[Span]:
        spans: list[Span] = []
        inner = SentenceChunker(self.config)
        for start, end, heading, level in self.sections(text):
            meta = {"heading": heading, "heading_level": level} if heading else {}
            if end - start <= self.config.chunk_size * 1.25:
                spans.append(Span(start, end, meta))
            else:
                spans.extend(Span(start + s.start, start + s.end, meta) for s in inner.split(text[start:end]))
        return spans
