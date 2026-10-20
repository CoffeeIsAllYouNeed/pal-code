"""One token budget for everything that goes into a request besides the user's own message.

History, the exact details of old turns, remembered notes, document excerpts and code-graph excerpts each have their own cap, and together
they could add up to a lot. The assembler keeps the total under `budget` by dropping the least useful first: remembered notes, then
the facts, then the oldest turns (down to the last two), then document excerpts, then code excerpts, then the remaining turns
(never the last one). What the user asks about right now (excerpts that matched it) outranks older history.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pal.context_manager.components.turns import Turn
from pal.tokens import estimate_tokens

# the order blocks are dropped in, least useful first
DROP_ORDER = ("memory", "facts", "documents", "graph")


@dataclass
class Block:
    source: str            # "facts" | "memory" | "documents" | "graph"
    text: str
    tokens: int = 0


@dataclass
class ContextReport:
    budget: int
    used: int
    by_source: dict = field(default_factory=dict)      # tokens per source, after fitting
    dropped: list = field(default_factory=list)        # block sources dropped to fit
    turns_dropped: int = 0
    tokens_dropped_from_turns: int = 0


class ContextAssembler:
    def __init__(self, budget: int, count: Callable[[str], int] = estimate_tokens):
        self.budget = budget
        self.count = count

    def fit(self, turns: list[Turn], blocks: list[Block], turn_cost: Callable[[Turn], int]) -> tuple[list[Turn], list[Block], ContextReport]:
        blocks = [Block(b.source, b.text, b.tokens or self.count(b.text)) for b in blocks]
        turns = list(turns)
        used = sum(turn_cost(t) for t in turns) + sum(b.tokens for b in blocks)
        dropped: list[str] = []
        lost = 0

        def drop_block(source: str) -> None:
            nonlocal used
            for b in [b for b in blocks if b.source == source]:
                blocks.remove(b)
                used -= b.tokens
                dropped.append(source)

        def drop_turns(keep: int) -> None:
            nonlocal used, lost
            while used > self.budget and len(turns) > keep:
                cost = turn_cost(turns.pop(0))
                used -= cost
                lost += cost

        before = len(turns)
        for source in ("memory", "facts"):
            if used > self.budget:
                drop_block(source)
        drop_turns(keep=2)
        for source in ("documents", "graph"):
            if used > self.budget:
                drop_block(source)
        drop_turns(keep=1)
        by_source = {"history": sum(turn_cost(t) for t in turns)}
        for b in blocks:
            by_source[b.source] = by_source.get(b.source, 0) + b.tokens
        return turns, blocks, ContextReport(self.budget, used, by_source, dropped, before - len(turns), lost)
