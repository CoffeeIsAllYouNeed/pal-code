import threading
import time
from types import SimpleNamespace

import pytest

from pal.llm_scheduler import ApiScheduler, CodeSlots, QueueFullError, SchedulerError, is_rate_limit


@pytest.fixture
def api():
    s = ApiScheduler(retries=2, backoff=0.02)
    yield s
    s.close()


def test_calls_run_strictly_in_arrival_order_one_at_a_time(api):
    spans = []

    def job(n):
        start = time.monotonic()
        time.sleep(0.03)
        spans.append((n, start, time.monotonic()))

    tickets = [api.submit("k", lambda n=n: job(n), label=str(n)) for n in range(6)]
    [t.result(5) for t in tickets]
    assert [s[0] for s in spans] == list(range(6))
    assert all(spans[i][2] <= spans[i + 1][1] for i in range(5))              # never two calls at once


def test_arrival_order_across_many_threads_is_the_service_order(api):
    arrived, served = [], []
    lock = threading.Lock()

    def caller(i):
        time.sleep(i * 0.03)
        with lock:
            arrived.append(i)
            t = api.submit("k", lambda: (time.sleep(0.04), served.append(i)), label=str(i))
        t.result(5)

    threads = [threading.Thread(target=caller, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert served == arrived == list(range(6))


def test_one_key_means_one_line_shared_by_everyone_and_different_keys_do_not_wait_for_each_other(api):
    peak, running = {"k1": 0, "all": 0}, {"n": 0}
    lock = threading.Lock()

    def job(key):
        with lock:
            running["n"] += 1
            peak["all"] = max(peak["all"], running["n"])
        time.sleep(0.05)
        with lock:
            running["n"] -= 1

    same = [api.submit("gemini:ab", lambda: job("k1")) for _ in range(4)]
    [t.result(5) for t in same]
    assert peak["all"] == 1                                                    # one key: strictly one at a time
    peak["all"] = 0
    two = [api.submit("gemini:aa", lambda: job("a")), api.submit("gemini:bb", lambda: job("b"))]
    [t.result(5) for t in two]
    assert peak["all"] == 2                                                    # two keys: two independent lines
    assert set(api.pending()) == {"gemini:ab", "gemini:aa", "gemini:bb"}


def test_calls_start_at_least_the_interval_apart(api):
    starts = []
    tickets = [api.submit("k", lambda: starts.append(time.monotonic()), interval=0.15) for _ in range(4)]
    [t.result(5) for t in tickets]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.14 for g in gaps)


def test_a_rate_limit_error_pauses_the_line_and_retries_the_same_call_before_anyone_behind_it(api):
    order, attempts = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
        order.append("flaky")
        return "ok"

    first = api.submit("k", flaky)
    second = api.submit("k", lambda: order.append("second"))
    assert first.result(5) == "ok"
    second.result(5)
    assert len(attempts) == 3 and order == ["flaky", "second"]                 # first come, first served even through retries


def test_retries_run_out_other_errors_are_not_retried_and_errors_are_the_original(api):
    calls = []

    def always():
        calls.append(1)
        raise RuntimeError("429 quota")

    with pytest.raises(RuntimeError, match="429"):
        api.submit("k", always).result(5)
    assert len(calls) == 3                                                      # one call and two retries
    other = []

    def broken():
        other.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        api.submit("k", broken).result(5)
    assert len(other) == 1
    assert api.submit("k", lambda: "still works").result(5) == "still works"


@pytest.mark.parametrize("error,expected", [
    (RuntimeError("429 Too Many Requests"), True), (RuntimeError("RESOURCE_EXHAUSTED"), True),
    (RuntimeError("quota exceeded for this key"), True), (RuntimeError("rate limit reached"), True),
    (type("E", (Exception,), {"status_code": 429})("x"), True),
    (type("E", (Exception,), {"code": 429})("x"), True),
    (type("E", (Exception,), {"response": SimpleNamespace(status_code=429)})("x"), True),
    (ValueError("bad request"), False), (RuntimeError("500 server error"), False)])
def test_rate_limit_detection(error, expected):
    assert is_rate_limit(error) is expected


def test_a_waiting_call_reports_how_long_it_waited(api):
    a = api.submit("k", lambda: time.sleep(0.12))
    b = api.submit("k", lambda: None)
    a.result(5)
    b.result(5)
    assert a.waited < 0.05 and b.waited >= 0.1 and b.run_seconds < 0.05


def test_the_line_has_a_limit_and_closing_fails_waiting_calls():
    s = ApiScheduler(max_queue=1)
    gate = threading.Event()
    s.submit("k", lambda: gate.wait(5))
    time.sleep(0.05)
    waiting = s.submit("k", lambda: 1)
    with pytest.raises(QueueFullError, match="too many API calls"):
        s.submit("k", lambda: 2)
    s.close()
    with pytest.raises(SchedulerError, match="closed"):
        waiting.result(5)
    gate.set()


def test_with_scheduling_off_the_call_runs_right_here_with_no_line():
    s = ApiScheduler(enabled=False)
    me = threading.get_ident()
    assert s.submit("k", threading.get_ident).result() == me and s.pending() == {}
    with pytest.raises(KeyError):
        s.submit("k", lambda: {}["x"]).result()


def test_code_work_runs_a_few_at_a_time_and_unlimited_when_off():
    slots = CodeSlots(2)
    done = []
    ts = [threading.Thread(target=lambda: done.append(slots.run(lambda: time.sleep(0.05) or 1))) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert slots.peak == 2 and done == [1] * 6 and slots.running == 0
    assert CodeSlots(0).run(lambda: "direct") == "direct"
    with pytest.raises(ZeroDivisionError):
        CodeSlots(1).run(lambda: 1 / 0)
