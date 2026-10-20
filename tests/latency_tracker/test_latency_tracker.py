"""The Latency Tracker: timing a turn in stages, and saying where the time went."""
import time

import pytest

from pal.latency_tracker import STAGES, LatencyTracker, percentile
from pal.metrics.components.collector import TurnMetric


def turn(route="api", total=10.0, stages=None, **kw):
    m = TurnMetric(route=route, latency_total=total, **kw)
    m.stage_seconds = dict(stages or {})
    return m


def test_a_stage_times_its_block_and_adds_up_when_entered_more_than_once():
    m = TurnMetric(route="api")
    for _ in range(2):
        with LatencyTracker.stage(m, "routing"):
            time.sleep(0.02)
    assert 0.04 <= m.stage_seconds["routing"] < 0.5


def test_a_stage_is_recorded_even_when_the_block_raises_and_an_unknown_stage_is_refused():
    m = TurnMetric(route="api")
    with pytest.raises(RuntimeError), LatencyTracker.stage(m, "context"):
        time.sleep(0.01)
        raise RuntimeError("boom")
    assert m.stage_seconds["context"] > 0
    with pytest.raises(KeyError, match="unknown stage 'thinking'"), LatencyTracker.stage(m, "thinking"):
        pass


def test_finish_adds_the_api_and_tool_times_measured_elsewhere_and_the_remainder_as_other():
    m = turn(total=10.0, stages={"routing": 1.0, "input_saver": 2.0}, latency_api=5.0, latency_tools=0.5)
    LatencyTracker.finish(m)
    assert m.stage_seconds == {"routing": 1.0, "input_saver": 2.0, "api": 5.0, "tools": 0.5, "other": 1.5}
    over = turn(total=1.0, stages={"context": 3.0})
    LatencyTracker.finish(over)
    assert over.stage_seconds["other"] == 0.0                                   # never negative


def test_the_summary_says_how_long_each_stage_takes_its_share_and_which_is_slowest():
    a = turn(total=10.0, stages={"input_saver": 3.0, "api": 6.0, "other": 1.0})
    b = turn(total=20.0, stages={"input_saver": 5.0, "api": 14.0, "other": 1.0})
    s = LatencyTracker.summarize([a, b])
    st = s["latency_stages"]
    assert st["api"] == {"turns": 2, "mean_sec": 10.0, "p95_sec": 14.0, "total_sec": 20.0, "share_pct": 66.7}
    assert st["input_saver"]["share_pct"] == 26.7 and "routing" not in st and s["slowest_stage"] == "api"


def test_the_summary_keeps_the_per_route_waiting_and_pause_numbers_it_always_had():
    t1 = turn("api", 10.0, latency_local=2.0, wait_local=1.0, wait_api=3.0, local_pauses=1)
    t2 = turn("local", 2.0)
    t3 = turn("tool", 0.1)
    s = LatencyTracker.summarize([t1, t2, t3])
    assert s["by_route"]["api"] == {"turns": 1, "avg_latency_sec": 10.0, "p50_latency_sec": 10.0, "p95_latency_sec": 10.0}
    assert s["by_route"]["local"]["turns"] == 1 and s["by_route"]["tool"]["avg_latency_sec"] == 0.1
    assert s["avg_local_overhead_sec_on_api_turns"] == 2.0 and s["avg_wait_local_sec"] == 0.333 and s["avg_wait_api_sec"] == 3.0 and s["local_pauses"] == 1


def test_an_empty_list_of_turns_and_the_percentile_helper():
    s = LatencyTracker.summarize([])
    assert s["latency_stages"] == {} and s["slowest_stage"] == "" and s["by_route"]["api"]["turns"] == 0
    assert percentile([], 0.5) == 0.0 and percentile([1, 2, 3, 4, 5], 0.5) == 3 and percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert STAGES[-1] == "other" and "input_saver" in STAGES
