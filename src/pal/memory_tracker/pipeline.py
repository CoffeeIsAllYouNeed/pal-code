"""The Memory Tracker: how much memory is the system using, and where?

Components that hold something (the Notes Manager's notes, the open sessions, the knowledge graph's projects, the storage folder) register a
gauge here: a small function that reports their numbers. `snapshot()` calls them all and adds the process's own memory, so one look shows
what is using memory and how full the caches are. A gauge that fails is reported as an error, never raised: tracking must not break the chat.
It also counts how many remembered notes were recalled into each turn.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pal.memory_tracker.components.process import dir_bytes, process_memory


class MemoryTracker:
    def __init__(self) -> None:
        self._gauges: dict[str, Callable[[], Any]] = {}

    def register(self, name: str, gauge: Callable[[], Any]) -> None:
        """Add (or replace) a gauge: a no-argument function returning a number or a dict of numbers."""
        self._gauges[name] = gauge

    def snapshot(self) -> dict:
        out: dict = {}
        for name, gauge in self._gauges.items():
            try:
                out[name] = gauge()
            except Exception as e:  # noqa: BLE001
                out[name] = {"error": type(e).__name__}
        out["process"] = process_memory()
        return out

    @staticmethod
    def note_hits(turn, hits: int) -> None:
        """Record how many remembered notes were recalled into this turn's request."""
        turn.memory_hits = int(hits)

    @staticmethod
    def summarize(turns: Iterable) -> dict:
        return {"memory_hits": sum(t.memory_hits for t in turns)}


__all__ = ["MemoryTracker", "dir_bytes", "process_memory"]
