from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from pal.latency_tracker import LatencyTracker
from pal.memory_tracker import MemoryTracker
from pal.token_tracker import TokenTracker


@dataclass
class RequestMetric:
    """Legacy per-request record used by the older Pipeline/experiments."""

    timestamp: float
    tokens_naive: int
    tokens_optimized: int
    latency: float
    cache_hit: bool

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_naive - self.tokens_optimized)


@dataclass
class TurnMetric:
    """One chat turn. Latencies are seconds. `saved_*` are estimates (chars/4 unless the provider
    counter is on): what a naive client would have sent minus what was sent."""

    route: str                      # "api" | "local" | "tool"
    session: str = ""
    provider: str = ""
    model: str = ""
    latency_total: float = 0.0
    latency_local: float = 0.0      # routing, input savers, local answer
    latency_api: float = 0.0        # API calls, tool rounds included
    latency_tools: float = 0.0
    input_tokens: int = 0           # billed by the API (0 for local/tool turns)
    output_tokens: int = 0
    local_tokens: int = 0
    saved_savers: int = 0
    saved_paging: int = 0
    saved_graph: int = 0
    saved_documents: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    memory_hits: int = 0
    escalated: bool = False
    wait_local: float = 0.0           # seconds this turn's local jobs waited in the local scheduler
    wait_api: float = 0.0             # seconds its API calls waited in the API line
    local_pauses: int = 0             # times a higher-priority job paused one of its local jobs
    compressed_turns: int = 0         # old assistant replies shown compressed this turn
    compression_saved: int = 0        # tokens that compression saved versus showing those turns in full
    cost_usd: float = 0.0             # what the API calls cost (LiteLLM's price table); 0 when the model has no known price
    saved_cost_usd: float = 0.0       # estimated tokens saved x the model's input price
    fallback: str = ""                # "provider/model" when the fallback model answered
    fallback_reason: str = ""         # what failed first, e.g. RateLimitError
    stage_seconds: dict = field(default_factory=dict)   # where this turn's time went, by stage (see latency_tracker)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def saved_total(self) -> int:
        return self.saved_savers + self.saved_paging + self.saved_graph + self.saved_documents


class MetricsCollector:
    """Thread-safe, bounded: keeps the most recent `max_records` records of each kind."""

    def __init__(self, max_records: int = 1000, tokens: TokenTracker | None = None, latency: LatencyTracker | None = None,
                 memory: MemoryTracker | None = None):
        self.tokens, self.latency, self.memory = tokens or TokenTracker(), latency or LatencyTracker(), memory or MemoryTracker()
        self._lock = threading.Lock()
        self.records: deque[RequestMetric] = deque(maxlen=max_records)
        self.turns: deque[TurnMetric] = deque(maxlen=max_records)
        self.total_turns = 0

    def record(self, tokens_naive: int, tokens_optimized: int, latency: float, cache_hit: bool):
        with self._lock:
            self.records.append(RequestMetric(time.time(), tokens_naive, tokens_optimized, latency, cache_hit))

    def record_turn(self, turn: TurnMetric) -> None:
        with self._lock:
            self.turns.append(turn)
            self.total_turns += 1

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self.turns.clear()
            self.total_turns = 0

    def snapshot(self) -> tuple[list[RequestMetric], list[TurnMetric]]:
        with self._lock:
            return list(self.records), list(self.turns)

    def summary(self, cache_overhead_bytes: int = 0) -> dict:
        records, _ = self.snapshot()
        if not records:
            return {"requests": 0}
        total_saved = sum(r.tokens_saved for r in records)
        avg_latency = sum(r.latency for r in records) / len(records)
        hit_rate = sum(1 for r in records if r.cache_hit) / len(records)
        # ~4 bytes per token as a rough common scale for cache overhead vs tokens saved
        return {
            "requests": len(records),
            "total_tokens_saved": total_saved,
            "avg_latency_sec": round(avg_latency, 4),
            "cache_hit_rate": round(hit_rate, 4),
            "cache_overhead_bytes": cache_overhead_bytes,
            "cache_pays_for_itself": cache_overhead_bytes <= total_saved * 4,
        }

    def chat_summary(self, session: str | None = None) -> dict:
        """The turns added up, by the three trackers: tokens (and savings, cost), latency (and where the time went), memory."""
        _, turns = self.snapshot()
        if session:
            turns = [t for t in turns if t.session == session[:8]]
        if not turns:
            return {"turns": 0}
        api_turns = [t for t in turns if t.route == "api"]
        return {
            "turns": len(turns),
            "turns_recorded": self.total_turns,
            "api_calls_avoided": len(turns) - len(api_turns),
            **self.tokens.summarize(turns),
            **self.latency.summarize(turns),
            **self.memory.summarize(turns),
            "fallbacks": sum(1 for t in turns if t.fallback),
            "tool_calls": sum(t.tool_calls for t in turns),
            "escalations": sum(1 for t in turns if t.escalated),
            "errors": sum(1 for t in turns if t.error),
        }

    def turn_dicts(self) -> list[dict]:
        _, turns = self.snapshot()
        return [{**asdict(t), "saved_total": t.saved_total} for t in turns]
