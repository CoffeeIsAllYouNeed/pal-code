"""The Latency Tracker: where does the time of a chat turn go?

Each turn is timed in stages, so a slow turn can be explained rather than guessed at:

    direct        a question answered by code (arithmetic, the time), no model
    routing       the local model deciding who answers
    local_answer  the local model answering
    input_saver   the local model trimming what was typed (and the check that it lost nothing)
    context       gathering the request: history, notes, document and code excerpts
    api           the API model, with its wait in the queue
    tools         tool rounds (a tool call and the model call that follows it)
    other         everything else (the turn's total minus the above)

`stage()` times a block into the turn's record; `finish()` adds the stages that were already measured elsewhere and the remainder;
`summarize()` turns a list of turns into per-route latency, queue waits, pauses, and each stage's average, 95th percentile and share of
all the time spent.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import contextmanager

STAGES = ("direct", "routing", "local_answer", "input_saver", "context", "api", "tools", "other")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


class LatencyTracker:
    @staticmethod
    @contextmanager
    def stage(turn, name: str):
        """Time the block and add it to the turn's `stage_seconds[name]` (a stage can be entered more than once in a turn)."""
        if name not in STAGES:
            raise KeyError(f"unknown stage {name!r} (known: {', '.join(STAGES)})")
        from pal import steps
        start = time.perf_counter()
        try:
            if name in ("routing", "local_answer"):           # which model answers is never shown
                yield
            else:
                with steps.step(name):                          # pal cli shows each stage as it runs
                    yield
        finally:
            turn.stage_seconds[name] = turn.stage_seconds.get(name, 0.0) + (time.perf_counter() - start)

    @staticmethod
    def finish(turn) -> None:
        """Close the turn's record: the API and tool times measured elsewhere, and `other` = the total minus every named stage."""
        st = turn.stage_seconds
        if "api" not in st and turn.latency_api:
            st["api"] = turn.latency_api
        if "tools" not in st and turn.latency_tools:
            st["tools"] = turn.latency_tools
        named = sum(v for k, v in st.items() if k != "other")
        st["other"] = max(0.0, turn.latency_total - named)
        for k in list(st):
            st[k] = round(st[k], 4)

    @staticmethod
    def summarize(turns: Iterable) -> dict:
        turns = list(turns)
        n = len(turns)
        by_route: dict = {}
        for r in ("api", "local", "tool"):
            lat = [t.latency_total for t in turns if t.route == r]
            by_route[r] = {"turns": len(lat), "avg_latency_sec": round(sum(lat) / len(lat), 3) if lat else 0.0,
                           "p50_latency_sec": round(percentile(lat, 0.5), 3), "p95_latency_sec": round(percentile(lat, 0.95), 3)}
        api_turns = [t for t in turns if t.route == "api"]
        total_time = sum(t.latency_total for t in turns)
        stages: dict = {}
        for name in STAGES:
            vals = [t.stage_seconds[name] for t in turns if t.stage_seconds.get(name, 0) > 0]
            if vals:
                stages[name] = {"turns": len(vals), "mean_sec": round(sum(vals) / len(vals), 3), "p95_sec": round(percentile(vals, 0.95), 3),
                                "total_sec": round(sum(vals), 3), "share_pct": round(100.0 * sum(vals) / total_time, 1) if total_time else 0.0}
        slowest = max(stages, key=lambda k: stages[k]["total_sec"]) if stages else ""
        return {
            "by_route": by_route,
            "avg_local_overhead_sec_on_api_turns": round(sum(t.latency_local for t in api_turns) / len(api_turns), 3) if api_turns else 0.0,
            "avg_wait_local_sec": round(sum(t.wait_local for t in turns) / n, 3) if n else 0.0,
            "avg_wait_api_sec": round(sum(t.wait_api for t in api_turns) / len(api_turns), 3) if api_turns else 0.0,
            "local_pauses": sum(t.local_pauses for t in turns),
            "latency_stages": stages,
            "slowest_stage": slowest,
        }
