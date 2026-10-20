"""Cuts source code at function and class boundaries. Python is parsed properly; other languages are cut at blank lines."""
from __future__ import annotations

import ast
import re

from localrag.ingest.chunkers.base import Chunker, Span
from localrag.ingest.chunkers.fixed import FixedChunker


def line_offsets(text: str) -> list[int]:
    """The character offset at which each line starts (index 0 is line 1)."""
    offsets, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        offsets.append(pos)
    return offsets


class CodeChunker(Chunker):
    name = "code"

    def split(self, text: str) -> list[Span]:
        try:
            spans = self._python_spans(text)
        except SyntaxError:
            spans = self._block_spans(text)
        out: list[Span] = []
        fixed = FixedChunker(self.config)
        for s in spans:
            if s.end - s.start > self.config.chunk_size * 2:        # a very long function: cut it into windows, keep its name
                out.extend(Span(s.start + p.start, s.start + p.end, s.meta) for p in fixed.split(text[s.start:s.end]))
            else:
                out.append(s)
        return out

    def _python_spans(self, text: str) -> list[Span]:
        tree = ast.parse(text)
        offs = line_offsets(text)
        spans: list[Span] = []
        pending_start: int | None = None
        for node in tree.body:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            start, end = offs[first - 1], offs[min(node.end_lineno or first, len(offs) - 1)]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if pending_start is not None:
                    spans.append(Span(pending_start, start, {"symbol": "module header"}))
                    pending_start = None
                spans.append(Span(start, end, {"symbol": node.name, "kind": type(node).__name__.replace("Def", "").lower()}))
            elif pending_start is None:
                pending_start = start
        if pending_start is not None:
            spans.append(Span(pending_start, len(text), {"symbol": "module body"}))
        return spans or [Span(0, len(text))]

    def _block_spans(self, text: str) -> list[Span]:
        spans, start = [], 0
        for m in re.finditer(r"\n\s*\n", text):
            if m.start() - start >= self.config.chunk_size // 2:
                spans.append(Span(start, m.start()))
                start = m.end()
        spans.append(Span(start, len(text)))
        return spans
