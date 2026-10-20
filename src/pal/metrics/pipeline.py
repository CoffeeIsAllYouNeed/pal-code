from __future__ import annotations

import logging
import os

from pal.metrics.components.collector import MetricsCollector, TurnMetric
from pal.metrics.components.exporter import export_json

log = logging.getLogger(__name__)


class MetricsPipeline:
    def __init__(self, max_records: int = 1000, enabled: bool = True):
        self.enabled = enabled
        self.collector = MetricsCollector(max_records=max_records)
        self.tokens, self.latency, self.memory = self.collector.tokens, self.collector.latency, self.collector.memory     # the three trackers

    # legacy per-request API (older Pipeline and experiments)
    def record(self, tokens_naive: int, tokens_optimized: int, latency: float, cache_hit: bool) -> None:
        self.collector.record(tokens_naive, tokens_optimized, latency, cache_hit)

    def record_turn(self, turn: TurnMetric) -> None:
        if not self.enabled:
            return
        self.collector.record_turn(turn)
        log.info(
            "turn route=%s provider=%s model=%s total=%.2fs local=%.2fs api=%.2fs in=%d out=%d saved~%d tools=%d%s",
            turn.route, turn.provider or "-", turn.model or "-", turn.latency_total, turn.latency_local,
            turn.latency_api, turn.input_tokens, turn.output_tokens, turn.saved_total, turn.tool_calls,
            f" error={turn.error}" if turn.error else "")

    def summary(self, cache_overhead_bytes: int = 0) -> dict:
        return self.collector.summary(cache_overhead_bytes=cache_overhead_bytes)

    def chat_summary(self, session: str | None = None) -> dict:
        return self.collector.chat_summary(session)

    def turns(self) -> list[dict]:
        return self.collector.turn_dicts()

    def reset(self) -> None:
        self.collector.reset()

    def export_json(self, path: str | os.PathLike) -> None:
        export_json(self.collector, path)

    @property
    def records(self):
        return self.collector.records
