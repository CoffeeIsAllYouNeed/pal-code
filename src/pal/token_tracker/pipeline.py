"""The Token Tracker: counts tokens and keeps the books on them.

  * COUNTING   `count()` is the one way to count tokens: a quick estimate (characters / 4), or the provider's own counter when the
               settings ask for it (`counter_for`).
  * SAVINGS    each part that saves tokens (the input savers, history paging, the knowledge graph, document excerpts) credits its saving
               to the turn through `credit()`, instead of every part reaching into the turn's record itself.
  * SUMMARY    `summarize()` adds a list of turns up: tokens billed, tokens saved by which part, what the API calls cost, what the savings
               would have cost.
All saved-token figures are estimates: what a naive client would have sent minus what was sent.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from pal.tokens import estimate_tokens, make_counter

# the name each part credits under  ->  the field on the turn's record
SOURCES = {"input_savers": "saved_savers", "context_paging": "saved_paging", "code_graph": "saved_graph", "documents": "saved_documents"}


class TokenTracker:
    def __init__(self, counter: Callable[[str], int] = estimate_tokens):
        self.counter = counter

    def count(self, text: str) -> int:
        return self.counter(text)

    @staticmethod
    def counter_for(mode: str, core=None) -> Callable[[str], int]:
        """The counter the settings ask for: the provider's own (`provider`), or the quick estimate."""
        return make_counter(mode, core)

    @staticmethod
    def credit(turn, source: str, tokens: int) -> None:
        """Record that `source` saved `tokens` on this turn. Unknown sources are an error: a typo must not silently lose a saving."""
        if source not in SOURCES:
            raise KeyError(f"unknown saving source {source!r} (known: {', '.join(SOURCES)})")
        setattr(turn, SOURCES[source], max(0, int(tokens)))

    @staticmethod
    def summarize(turns: Iterable) -> dict:
        turns = list(turns)
        saved = {name: sum(getattr(t, field) for t in turns) for name, field in SOURCES.items()}
        saved["total"] = sum(t.saved_total for t in turns)
        return {
            "api_input_tokens": sum(t.input_tokens for t in turns),
            "api_output_tokens": sum(t.output_tokens for t in turns),
            "local_llm_tokens": sum(t.local_tokens for t in turns),
            "tokens_saved_estimate": saved,
            "compressed_turns": sum(t.compressed_turns for t in turns),
            "compression_saved": sum(t.compression_saved for t in turns),
            "cost_usd": round(sum(t.cost_usd for t in turns), 6),
            "saved_cost_usd": round(sum(t.saved_cost_usd for t in turns), 6),
        }
