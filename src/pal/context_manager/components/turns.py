"""Paging for chat history, one turn (user message + reply) at a time.

The last `hot_turns` turns are always sent. Older turns are "cold": they are dropped from the
request unless they look relevant to the current question, in which case up to `recall_turns` of
them are paged back in (best match first, sent in chronological order). A token cap on the hot
window trims it further if a few turns are very long.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pal.context_manager.components.pager import ContextPager, Segment
from pal.tokens import estimate_tokens


@dataclass
class Turn:
    user: str
    assistant: str
    local: bool = False                       # answered by the local LLM
    ts: float = field(default_factory=time.time)
    id: str = ""
    files: tuple = ()                         # files an agent saved in this turn (the content lives in the workspace)
    # cached by the compressor, never part of the turn's identity
    _keys: tuple[frozenset, frozenset, list] | None = field(default=None, repr=False, compare=False)
    _facts: tuple[list[str], list[str]] | None = field(default=None, repr=False, compare=False)

    def text(self) -> str:
        return f"{self.user}\n{self.assistant}"


@dataclass
class PagedHistory:
    turns: list[Turn]             # what to send, chronological
    recalled: int                 # cold turns paged back in
    full_tokens: int              # cost of sending every turn
    sent_tokens: int              # cost of what is sent
    dropped_turns: int

    @property
    def saved_tokens(self) -> int:
        return max(0, self.full_tokens - self.sent_tokens)


class TurnPager:
    def __init__(self, hot_turns: int = 6, recall_turns: int = 2, recall_threshold: float = 0.35,
                 max_history_tokens: int = 6000, count: Callable[[str], int] = estimate_tokens):
        self.hot_turns = hot_turns
        self.recall_turns = recall_turns
        self.recall_threshold = recall_threshold
        self.max_history_tokens = max_history_tokens
        self.count = count

    def _cost(self, turn: Turn) -> int:
        return self.count(turn.user) + self.count(turn.assistant) + 8

    turn_cost = _cost     # public name for the assembler

    def select(self, turns: list[Turn], query: str) -> PagedHistory:
        full = sum(self._cost(t) for t in turns)
        if len(turns) <= 1 and full <= self.max_history_tokens:
            return PagedHistory(list(turns), 0, full, full, 0)

        index = {str(i): t for i, t in enumerate(turns)}
        segments = [Segment(id=str(i), content=t.text(), role="turn", timestamp=t.ts) for i, t in enumerate(turns)]
        pager = ContextPager(hot_window=self.hot_turns, cold_threshold=self.recall_threshold)
        hot_segs, cold_refs = pager.page_out(segments)
        hot = [index[s.id] for s in hot_segs]

        # token cap: push the oldest hot turns to cold while over budget (always keep the newest)
        while len(hot) > 1 and sum(self._cost(t) for t in hot) > self.max_history_tokens:
            oldest = hot.pop(0)
            cold_refs.append(next(k for k, v in index.items() if v is oldest))

        recalled: list[Turn] = []
        if self.recall_turns and cold_refs and query.strip():
            room = self.max_history_tokens - sum(self._cost(t) for t in hot)
            for seg in pager.page_in(query, cold_refs, top_k=self.recall_turns):
                t = index[seg.id]
                if self._cost(t) <= room:
                    recalled.append(t)
                    room -= self._cost(t)
        pos = {id(t): i for i, t in enumerate(turns)}
        chosen = sorted(recalled + hot, key=lambda t: pos[id(t)])
        sent = sum(self._cost(t) for t in chosen)
        return PagedHistory(chosen, len(recalled), full, sent, len(turns) - len(chosen))
