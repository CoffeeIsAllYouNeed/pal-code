"""The Memory Tracker: gauges from the components that hold something, and the process's own memory."""
from pal.memory_tracker import MemoryTracker, dir_bytes, process_memory
from pal.metrics.components.collector import TurnMetric


def test_a_snapshot_calls_every_registered_gauge_and_adds_the_process_memory():
    mt = MemoryTracker()
    mt.register("notes", lambda: {"notes": 3, "ram_count": 2})
    mt.register("sessions", lambda: 5)
    snap = mt.snapshot()
    assert snap["notes"] == {"notes": 3, "ram_count": 2} and snap["sessions"] == 5 and set(snap["process"]) == {"rss_mb", "peak_rss_mb", "source"}


def test_a_gauge_that_fails_is_reported_not_raised_and_a_gauge_can_be_replaced():
    mt = MemoryTracker()
    mt.register("broken", lambda: 1 / 0)
    mt.register("ok", lambda: 1)
    snap = mt.snapshot()
    assert snap["broken"] == {"error": "ZeroDivisionError"} and snap["ok"] == 1
    mt.register("broken", lambda: 2)
    assert mt.snapshot()["broken"] == 2


def test_recalled_notes_are_counted_per_turn_and_added_up():
    a, b = TurnMetric(route="api"), TurnMetric(route="api")
    MemoryTracker.note_hits(a, 3)
    MemoryTracker.note_hits(b, 2)
    assert a.memory_hits == 3 and MemoryTracker.summarize([a, b]) == {"memory_hits": 5} and MemoryTracker.summarize([]) == {"memory_hits": 0}


def test_the_process_memory_reading_never_raises_and_says_where_it_came_from():
    p = process_memory()
    assert p["source"] in ("psutil", "resource", "win32", "unavailable")
    assert p["source"] == "unavailable" or (p["rss_mb"] or p["peak_rss_mb"]) > 0


def test_dir_bytes_adds_up_file_sizes_and_stops_counting_at_the_limit(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"1" * 100)
    (tmp_path / "y.bin").write_bytes(b"1" * 50)
    assert dir_bytes(tmp_path) == 150 and dir_bytes(tmp_path / "nothing") == 0
    for i in range(5):
        (tmp_path / f"f{i}").write_bytes(b"1" * 10)
    assert dir_bytes(tmp_path, limit_files=3) < dir_bytes(tmp_path)
