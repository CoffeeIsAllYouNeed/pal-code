"""Citations: for every hit, where in which source it came from and a short snippet around the words that matched."""
from __future__ import annotations

from localrag.core.textutil import tokenize
from localrag.core.types import Citation, SearchHit


def make_snippet(text: str, query_tokens: list[str], width: int = 160) -> str:
    """A window of about `width` characters centred on the first query word found in the text (the start of the text if none is)."""
    flat = " ".join(text.split())
    lower = flat.lower()
    positions = [i for t in query_tokens for i in [lower.find(t[:max(3, len(t) - 2)])] if i >= 0]
    centre = min(positions) if positions else 0
    start = max(0, min(centre - width // 4, len(flat) - width))
    piece = flat[start:start + width].strip()
    return ("... " if start > 0 else "") + piece + (" ..." if start + width < len(flat) else "")


def build_citations(hits: list[SearchHit], query: str, width: int = 160) -> list[Citation]:
    """One citation per hit, in rank order."""
    tokens = tokenize(query, stem=False)
    return [Citation(h.chunk.doc_id, h.chunk.id, str(h.chunk.metadata.get("source", h.chunk.doc_id)), h.chunk.start, h.chunk.end,
                     make_snippet(h.chunk.text, tokens, width)) for h in hits]
