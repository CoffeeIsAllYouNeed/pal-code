"""The context block handed to a model: numbered evidence passages that fit a token budget, each labelled with its source."""
from __future__ import annotations

from localrag.core.textutil import estimate_tokens, truncate_to_tokens
from localrag.core.types import SearchHit


def label(hit: SearchHit, number: int) -> str:
    """The heading line of one passage: its number, its source and, when it has one, its section or symbol."""
    meta = hit.chunk.metadata
    where = meta.get("heading") or meta.get("symbol") or ""
    return f"[{number}] {meta.get('source', hit.chunk.doc_id)}" + (f" - {where}" if where else "")


def build_context(hits: list[SearchHit], budget_tokens: int) -> str:
    """Numbered passages in rank order until the token budget is used up; the last passage is shortened rather than dropped."""
    blocks: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        head = label(hit, number)
        room = budget_tokens - used - estimate_tokens(head)
        if room <= 20 and blocks:
            break
        body = truncate_to_tokens(hit.chunk.text, max(room, 20))
        blocks.append(f"{head}\n{body}")
        used += estimate_tokens(head) + estimate_tokens(body)
        if used >= budget_tokens:
            break
    return "\n\n".join(blocks)
