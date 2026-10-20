"""The 30-experiment suite: logic only, with fake clients (no model, no Ollama)."""
from __future__ import annotations

import pytest
from experiments.demo import suite_run as s
from experiments.demo.suite_tasks import TASKS

from pal.terminal.clients import Reply


class Fake:
    def __init__(self, tokens):
        self.tokens, self.closed = tokens, False

    def ask(self, message, references=None):
        return Reply("ok", self.tokens, 5, 1, 0.0, 1)

    def close(self):
        self.closed = True


def test_thirty_varied_tasks():
    assert len(TASKS) == 30 and len({t.title for t in TASKS}) == 30
    assert len({t.category for t in TASKS}) >= 5
    assert all(len(t.steps()) >= 3 and t.spec_text() for t in TASKS)


def test_measure_and_filter():
    saved = s.measure(TASKS[0], lambda: Fake(500), lambda: Fake(200))
    assert saved["tokens_before"] > saved["tokens_after"] and saved["tokens_saved_pct"] > 0 and s.saved_tokens(saved)
    worse = s.measure(TASKS[1], lambda: Fake(100), lambda: Fake(300))
    assert not s.saved_tokens(worse)


def test_csv_only_has_saved_rows_and_charts(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / s.CSV_NAME
    for t, (a, b) in zip(TASKS[:3], [(500, 200), (100, 300), (400, 100)], strict=True):
        row = s.measure(t, lambda a=a: Fake(a), lambda b=b: Fake(b))
        row["model"] = "m"
        if s.saved_tokens(row):
            s.append_row(path, row)
    rows = s.read_rows(path)
    assert [r["id"] for r in rows] == ["1", "3"] and list(rows[0]) == s.COLUMNS
    charts = s.draw_charts(path, tmp_path)
    assert [p.name for p in charts] == ["suite_tokens.png", "suite_latency.png", "suite_memory.png"] and all(p.exists() for p in charts)


def test_only_parsing():
    assert s.parse_only("1,5,12-14") == {1, 5, 12, 13, 14}
    with pytest.raises(ValueError):
        s.parse_only("31")
