"""The context manager for chat: it decides everything that goes into a request besides the user's own message, under one budget.

What it keeps for itself: choosing which turns of the history are shown (the pager: the newest verbatim, relevant older ones recalled),
and fitting everything under `context.total_budget_tokens` with the assembler, dropping the least useful first.
What it asks other components for:
  * the Compression component      shortens the older replies and turns the oldest turns into facts (compression/history.py);
  * the Knowledge Graph            the excerpts of an uploaded project that match the question (knowledge_graph);
  * the Notes Manager              the notes from other sessions that match the question (notes_manager);
  * the documents index            the parts of attached documents that match.
It does no background work: the tidy-up of notes belongs to the Notes Manager.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pal.compression.history import MARK, HistoryCompressor, composition
from pal.config import ContextConfig
from pal.context_manager.components.assembler import Block, ContextAssembler, ContextReport
from pal.context_manager.components.turns import PagedHistory, Turn, TurnPager
from pal.memory_tracker import MemoryTracker
from pal.token_tracker import TokenTracker
from pal.tokens import estimate_tokens

log = logging.getLogger(__name__)


@dataclass
class ContextBundle:
    turns: list[Turn]                    # what goes into the request, chronological (some are compressed copies)
    blocks: list[Block]                  # kept, in the order they go into the request
    report: ContextReport
    paged: PagedHistory | None
    recall: Any
    doc_info: dict | None
    graph_info: dict | None
    compressed: int = 0                  # compressed turns that were kept
    compression_saved: int = 0           # tokens saved versus showing those turns verbatim
    cold_turns: int = 0                  # turns shown only as facts

    @property
    def texts(self) -> list[str]:
        return [b.text for b in self.blocks]

    @property
    def uses_compression(self) -> bool:
        return self.compressed > 0 or any(b.source == "facts" for b in self.blocks)


class ChatContext:
    def __init__(self, cfg: ContextConfig, pager: TurnPager, notes, graph,
                 local_inner: Callable[[], Any] | None = None, count: Callable[[str], int] = estimate_tokens,
                 tokens: TokenTracker | None = None, memory_tracker: MemoryTracker | None = None):
        self.cfg, self.pager, self.notes, self.graph, self.count = cfg, pager, notes, graph, count
        self.tokens, self.memory_tracker = tokens or TokenTracker(count), memory_tracker or MemoryTracker()
        self.assembler = ContextAssembler(cfg.total_budget_tokens, count)
        self.compression = HistoryCompressor(cfg, pager.turn_cost, count)

    # -- gathering and fitting -------------------------------------------------------------------------------------
    def build(self, sess, sid: str, message: str, new_docs: set[str], project: bool, counter, ephemeral: bool, m) -> ContextBundle:
        all_turns: list[Turn] = sess.turns
        paged = self.pager.select(all_turns, message) if self.cfg.enabled else None
        selected = paged.turns if paged else list(all_turns)
        layered = self.compression.layer(all_turns, selected, paged)
        history, compact_ids, cold_idx = layered.history, layered.compact_ids, layered.cold_idx
        warm_verbatim, warm_compact = layered.warm_verbatim, layered.warm_compact

        blocks: list[Block] = []
        facts = self.compression.facts(all_turns, cold_idx)
        if facts:
            blocks.append(Block("facts", facts))

        recall = self.notes.recall(sid, message) if not ephemeral else None
        if recall:
            blocks.append(Block("memory", recall.text))
            self.memory_tracker.note_hits(m, recall.hits)

        doc_info = None
        if sess.docs is not None and any(n not in new_docs for n in sess.docs.docs):
            block, used, n_chunks = sess.docs.excerpts(message, exclude=new_docs)
            earlier = sum(d.tokens for nme, d in sess.docs.docs.items() if nme not in new_docs)
            self.tokens.credit(m, "documents", earlier - used)
            if block:
                blocks.append(Block("documents", block))
            doc_info = {"chunks": n_chunks, "tokens": used, "indexed_tokens": earlier}

        graph_info = None
        if project:
            gb = self.graph.context_block(sid, message, counter)
            if gb:
                blocks.append(Block("graph", gb.text))
                self.tokens.credit(m, "code_graph", gb.saved_tokens)
                graph_info = gb.info

        turns, kept, report = self.assembler.fit(history, blocks, self.pager.turn_cost)
        facts_kept = sum(b.tokens for b in kept if b.source == "facts")
        kept_compact = [t for t in turns if id(t) in compact_ids]
        self.tokens.credit(m, "context_paging", paged.full_tokens - sum(self.pager.turn_cost(t) for t in turns) - facts_kept if paged else 0)
        m.compressed_turns = len(kept_compact)
        saved = (warm_verbatim - warm_compact) if kept_compact else 0
        m.compression_saved = saved
        return ContextBundle(turns, kept, report, paged, recall, doc_info, graph_info, len(kept_compact), saved,
                             len(cold_idx) if facts_kept else 0)

    def describe(self, sess, bundle: ContextBundle, m) -> dict:
        r = bundle.report
        return {"turns_sent": len(bundle.turns), "turns_total": len(sess.turns) - 1,
                "recalled": bundle.paged.recalled if bundle.paged else 0, "tokens_saved": m.saved_paging,
                "budget": r.budget, "used": r.used, "by_source": r.by_source, "dropped": r.dropped,
                "compression": {"compressed_turns": bundle.compressed, "tokens_saved": bundle.compression_saved,
                                "facts_from_turns": bundle.cold_turns},
                "history_tokens": composition(sess.turns, self.count)}      # what is big in the whole conversation so far


__all__ = ["ChatContext", "ContextBundle", "MARK"]
