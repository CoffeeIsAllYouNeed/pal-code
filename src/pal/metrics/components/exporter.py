from __future__ import annotations

import json
import os
from pathlib import Path

from pal.metrics.components.collector import MetricsCollector


def export_json(collector: MetricsCollector, path: str | os.PathLike) -> None:
    records, _ = collector.snapshot()
    data = {
        "summary": collector.summary(),
        "chat": collector.chat_summary(),
        "turns": collector.turn_dicts(),
        "records": [
            {
                "timestamp": r.timestamp,
                "tokens_naive": r.tokens_naive,
                "tokens_optimized": r.tokens_optimized,
                "tokens_saved": r.tokens_saved,
                "latency": r.latency,
                "cache_hit": r.cache_hit,
            }
            for r in records
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
