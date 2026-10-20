import json
import threading

from pal.metrics import MetricsPipeline, TurnMetric


def turn(route="api", total=1.0, **kw):
    return TurnMetric(route=route, latency_total=total, **kw)


def test_empty_summary():
    assert MetricsPipeline().chat_summary() == {"turns": 0}


def test_summary_counts_routes_percentiles_and_savings():
    m = MetricsPipeline()
    for i in range(1, 101):
        m.record_turn(turn("api", total=i / 10, input_tokens=100, output_tokens=20, saved_savers=5, saved_paging=10,
                           saved_graph=1, saved_documents=2, latency_local=0.5, tool_calls=1, memory_hits=1))
    m.record_turn(turn("local", total=0.2))
    m.record_turn(turn("tool", total=0.001, escalated=False))
    m.record_turn(turn("api", error="ProviderError"))
    s = m.chat_summary()
    assert s["turns"] == 103 and s["api_calls_avoided"] == 2 and s["errors"] == 1
    api = s["by_route"]["api"]
    assert api["turns"] == 101 and 4.9 <= api["p50_latency_sec"] <= 5.2 and api["p95_latency_sec"] >= 9.4
    assert s["tokens_saved_estimate"] == {"input_savers": 500, "context_paging": 1000, "code_graph": 100,
                                          "documents": 200, "total": 1800}
    assert s["api_input_tokens"] == 10000 and s["tool_calls"] == 100 and s["memory_hits"] == 100


def test_records_are_bounded():
    m = MetricsPipeline(max_records=10)
    for _ in range(50):
        m.record_turn(turn())
    s = m.chat_summary()
    assert s["turns"] == 10 and s["turns_recorded"] == 50 and len(m.turns()) == 10


def test_disabled_pipeline_records_nothing():
    m = MetricsPipeline(enabled=False)
    m.record_turn(turn())
    assert m.chat_summary() == {"turns": 0}


def test_thread_safety():
    m = MetricsPipeline(max_records=100000)
    ts = [threading.Thread(target=lambda: [m.record_turn(turn()) for _ in range(500)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert m.chat_summary()["turns"] == 4000


def test_legacy_request_api_still_works_and_export_is_utf8(tmp_path):
    m = MetricsPipeline()
    m.record(100, 60, 0.5, True)
    m.record_turn(turn(session="é"))
    assert m.summary()["total_tokens_saved"] == 40
    out = tmp_path / "m.json"
    m.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["requests"] == 1 and data["turns"][0]["session"] == "é" and data["chat"]["turns"] == 1
