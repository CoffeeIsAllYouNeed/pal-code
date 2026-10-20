"""Recall: which notes from OTHER sessions go into one request, within a token budget."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Recall:
    text: str      # the <memory> block to add to the request
    hits: int      # how many notes are in it
    tokens: int    # what they cost


def build_recall(notes, session_id: str, limit: int, budget_tokens: int, count: Callable[[str], int]) -> Recall | None:
    """The best matching notes written by other sessions (this session's own messages are already in the history), at most `limit` of
    them and within `budget_tokens`. None when nothing fits."""
    lines: list[str] = []
    used = 0
    for n in notes:
        if n.session_id == session_id:
            continue
        line = f"- {n.content}"
        cost = count(line)
        if used + cost > budget_tokens or len(lines) >= limit:
            continue
        lines.append(line)
        used += cost
    if not lines:
        return None
    return Recall("<memory>\nEarlier messages from this user, from other sessions:\n" + "\n".join(lines) + "\n</memory>", len(lines), used)
