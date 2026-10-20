"""The scheduler inside real chat turns: the API model's single first-come-first-served line, the local model's priority
(agent, then chat, then other) with pause and resume, queue statistics, and the code slots."""
import io
import re
import threading
import time

import pytest

from pal.llm_scheduler import TaskClass
from pal.llm_wrapper.components.base import LLMResponse
from tests.fakes import FakeAPI, RouterLocal
from tests.helpers import client, make_service, send

A, C, BG = TaskClass.AGENT, TaskClass.CHAT, TaskClass.OTHER
KEY = "gemini-key-1"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_CODING", "GEMINI_API_KEY_DOCS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_CODING", KEY)


class SlowAPI(FakeAPI):
    """Records when each call starts, how many run at once, and which message it was for."""

    def __init__(self, delay=0.05, **kw):
        super().__init__(**kw)
        self.delay, self.active, self.peak, self.order = delay, 0, 0, []
        self._lock = threading.Lock()

    def chat(self, syscall):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(syscall.messages[-1]["content"])
        try:
            time.sleep(self.delay)
            return super().chat(syscall)
        finally:
            with self._lock:
                self.active -= 1


class OrderedLocal(RouterLocal):
    """Records every local-model call (what kind, for which message) in the order the model actually ran them."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.order = []

    def chat(self, syscall):
        if len(syscall.messages) == 1:                       # the warm-up ping
            self.order.append(("warm", syscall.messages[0]["content"]))
            return LLMResponse(text="hi")
        kind = "route" if syscall.messages[0]["content"].startswith("You route") else "trim"
        self.order.append((kind, syscall.messages[-1]["content"]))
        return super().chat(syscall)


def run_in_threads(*targets, gap=0.0):
    ts = [threading.Thread(target=t) for t in targets]
    for t in ts:
        t.start()
        time.sleep(gap)
    [t.join(15) for t in ts]


# -- API: one line, strictly first come first served -----------------------------------------------------------------
def test_every_api_call_waits_in_one_line_in_arrival_order(tmp_path):
    api = SlowAPI(0.06)
    svc, _ = make_service(tmp_path, api=api)
    run_in_threads(*[lambda i=i: send(svc, f"question {i} about widgets", route="api", sid=f"s{i}") for i in range(5)], gap=0.02)
    assert api.peak == 1                                                         # never two API calls at once
    assert [int(re.search(r"\d", m).group()) for m in api.order] == [0, 1, 2, 3, 4]   # served in the order they arrived
    assert len(svc.metrics.turns()) == 5


def test_chat_and_agents_share_one_line_when_they_share_the_key(tmp_path):
    api = SlowAPI(0.08)
    svc, _ = make_service(tmp_path, api=api, scheduler={"api_min_interval": {"gemini": 0.0}})
    seen = []

    def chat():
        seen.append(svc.send("c", "gemini", "m", KEY, "chat question about gadgets", [], route="api"))   # the same key

    def agent():
        seen.append(svc.send("a", "", "", None, "agent question about code", [], agent="coding"))

    run_in_threads(chat, agent, gap=0.03)
    assert len(seen) == 2 and api.peak == 1
    assert "gadgets" in api.order[0] and "code" in api.order[1]                  # chat arrived first, so it is served first


def test_the_second_caller_waits_and_the_wait_is_recorded(tmp_path):
    api = SlowAPI(0.3)
    svc, _ = make_service(tmp_path, api=api)
    run_in_threads(lambda: send(svc, "first question about widgets", route="api", sid="one"),
                   lambda: send(svc, "second question about gadgets", route="api", sid="two"), gap=0.1)
    by_session = {t["session"]: t for t in svc.metrics.turns()}
    assert by_session["one"]["wait_api"] < 0.1 and by_session["two"]["wait_api"] >= 0.15
    s = svc.metrics.chat_summary()
    assert s["avg_wait_api_sec"] >= 0.08 and "local_pauses" in s and "avg_wait_local_sec" in s


def test_gemini_calls_are_spaced_by_the_rate_limiter_other_providers_are_not(tmp_path):
    starts = []

    class Stamp(FakeAPI):
        def chat(self, syscall):
            starts.append(time.monotonic())
            return super().chat(syscall)

    svc, _ = make_service(tmp_path, api=Stamp(), scheduler={"api_min_interval": {"gemini": 0.25}})
    for i in range(2):
        svc.send(f"g{i}", "", "", None, f"explain the cache {i}", [], agent="coding")
    assert starts[1] - starts[0] >= 0.24
    starts.clear()
    for i in range(2):
        send(svc, f"explain the cache {i}", route="api", sid=f"p{i}")            # provider "anthropic": no spacing
    assert starts[1] - starts[0] < 0.2


def test_a_rate_limit_error_is_retried_by_the_line_and_the_user_never_sees_it(tmp_path):
    class FailsOnce(FakeAPI):
        def chat(self, syscall):
            if not self.seen:
                self.seen.append(syscall.messages)
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return super().chat(syscall)

    api = FailsOnce(["answer after retry"])
    svc, _ = make_service(tmp_path, api=api, scheduler={"api_backoff_seconds": 0.01})
    out = send(svc, "explain the cache", route="api")
    assert out.reply == "answer after retry" and len(api.seen) == 2


# -- local model: agent, then chat, then other ---------------------------------------------------------------------------
def test_an_agents_local_preparation_goes_before_a_chat_message_that_arrived_earlier(tmp_path):
    local = OrderedLocal()
    svc, _ = make_service(tmp_path, local=local, api=FakeAPI(["x"]), scheduler={"api_min_interval": {"gemini": 0.0}})
    started, release = threading.Event(), threading.Event()
    svc.scheduler.local.submit(lambda: (started.set(), release.wait(5)), BG, "background")
    started.wait(2)
    chat = threading.Thread(target=lambda: send(svc, "capital of france", sid="chat"))
    chat.start()
    while svc.scheduler.local.pending()["chat"] < 1:
        time.sleep(0.01)
    agent = threading.Thread(target=lambda: svc.send("agent", "", "", None, "please explain the caching layer", [], agent="coding"))
    agent.start()
    while svc.scheduler.local.pending()["agent"] < 1:
        time.sleep(0.01)
    release.set()
    chat.join(10)
    agent.join(10)
    assert local.order[0][0] == "trim" and "caching layer" in local.order[0][1]  # the agent's job ran first
    assert any(kind == "route" and "capital of france" in msg for kind, msg in local.order)   # the chat job ran too


def test_an_agent_never_asks_the_local_model_to_route_or_answer(tmp_path):
    local = OrderedLocal()
    svc, _ = make_service(tmp_path, local=local, api=FakeAPI(["x"]))
    svc.send("a", "", "", None, "please explain the caching layer", [], agent="coding")
    kinds = {k for k, _ in local.order}
    assert kinds == {"trim"} and local.route_reply and not local.classified and not local.answered


class StreamLocal(OrderedLocal):
    """A local model that really streams its answer, slowly, in chunks."""

    def __init__(self, svc_holder, **kw):
        super().__init__(route='{"route": "local", "reason": "easy"}', **kw)
        self.holder, self.stream_calls, self.fired = svc_holder, [], False

    def stream_chat(self, syscall):
        self.stream_calls.append(syscall.messages)
        resumed = syscall.messages[-1]["content"].startswith("[Continue")
        for chunk in (["the capital ", "of France."] if resumed else ["Paris ", "is ", "the capital ", "of France."]):
            if chunk == "is " and not self.fired:
                self.fired = True
                self.holder["svc"].scheduler.local.submit(lambda: None, A, "agent step")   # an agent job arrives mid-answer
            time.sleep(0.02)
            yield chunk


def test_a_local_answer_is_paused_for_an_agent_job_and_then_continues(tmp_path):
    holder = {}
    local = StreamLocal(holder)
    svc, api = make_service(tmp_path, local=local, api=FakeAPI(["unused"]))
    holder["svc"] = svc
    out = send(svc, "capital of france")
    assert out.handled_by == "local" and out.reply == "Paris is the capital of France."
    assert len(local.stream_calls) == 2 and local.stream_calls[1][-2]["role"] == "assistant"
    assert svc.metrics.turns()[-1]["local_pauses"] == 1 and api.seen == []


def test_agent_preparation_cannot_pause_atomic_helper_calls_it_waits_for_them(tmp_path):
    local = OrderedLocal()
    svc, _ = make_service(tmp_path, local=local, api=FakeAPI(["x"]))
    out = svc.send("a", "", "", None, "please explain the caching layer", [], agent="coding")
    assert out.saver.tokens_saved >= 0 and svc.metrics.turns()[-1]["local_pauses"] == 0


# -- background work -------------------------------------------------------------------------------------------------------------
def test_background_work_runs_last_and_the_warm_up_is_a_background_job(tmp_path):
    local = OrderedLocal()
    svc, _ = make_service(tmp_path, local=local, api=FakeAPI(["x"]))
    started, release = threading.Event(), threading.Event()
    svc.scheduler.local.submit(lambda: (started.set(), release.wait(5)), BG, "busy")
    started.wait(2)
    warm = svc.warm_up_local()
    other_done = []
    svc.scheduler.local.submit(lambda: other_done.append("agent job"), A, "agent step")
    assert warm.task_class == BG
    release.set()
    warm.result(5)
    assert other_done == ["agent job"] and svc.scheduler.local.history[-1]["label"] == "warm up the local model"
    assert local.order == [("warm", "hi")]                  # it reached the real model, as a one-token ping
    labels = [h["label"] for h in svc.scheduler.local.history]
    assert labels.index("agent step") < labels.index("warm up the local model")      # the agent job went first


# -- code work and switches ----------------------------------------------------------------------------------------------------------
def test_uploads_convert_and_index_through_the_code_slots_not_the_llm_line(tmp_path):
    svc, _ = make_service(tmp_path, scheduler={"code_workers": 1})
    used = []
    real = svc.scheduler.code.run
    svc.scheduler.code.run = lambda fn: used.append(1) or real(fn)
    local_jobs_before = len(svc.scheduler.local.history)
    r = client(svc).post("/api/chat", data={"session_id": "w", "message": "read it", "provider": "anthropic", "model": "m",
                                            "api_key": "k", "route": "api", "files": (io.BytesIO(b"plain notes"), "n.txt")})
    assert r.status_code == 200 and used == [1]
    assert all("convert" not in h["label"] for h in list(svc.scheduler.local.history)[local_jobs_before:])


def test_metrics_endpoint_includes_the_scheduler_snapshot(tmp_path):
    svc, _ = make_service(tmp_path)
    snap = client(svc).get("/api/metrics").get_json()["scheduler"]
    assert snap["enabled"] is True and snap["local_waiting"] == {"agent": 0, "chat": 0, "other": 0}


def test_with_scheduling_switched_off_turns_still_work_with_no_lines(tmp_path):
    api = SlowAPI(0.05)
    svc, _ = make_service(tmp_path, api=api, scheduler={"enabled": False})
    assert svc.scheduler.local is None
    run_in_threads(*[lambda i=i: send(svc, f"question {i} about widgets", route="api", sid=f"s{i}") for i in range(3)], gap=0.0)
    assert api.peak >= 2                                                         # no line: calls overlap
    assert svc.metrics.chat_summary()["local_pauses"] == 0


def test_closing_the_service_stops_the_scheduler(tmp_path):
    svc, _ = make_service(tmp_path)
    send(svc, "explain the cache", route="api")
    svc.close()
    with pytest.raises(Exception, match="closed"):
        svc.scheduler.local.submit(lambda: 1)


def test_scheduler_settings_are_validated():
    from pal.config import load_settings
    from pal.errors import ConfigError
    for bad, msg in [({"local_max_pauses": -1}, ">= 0"), ({"max_queue": 0}, ">= 1"), ({"other_stale_seconds": 0}, "> 0"),
                     ({"api_min_interval": {"gemini": -1}}, "provider names to seconds"),
                     ({"api_min_interval": {"gemini": "fast"}}, "provider names to seconds"), ({"enabled": "yes"}, "true/false")]:
        with pytest.raises(ConfigError, match=msg):
            load_settings(env={}, overrides={"scheduler": bad})
    s = load_settings(env={}, overrides={"scheduler": {"api_min_interval": {"openai": 1}}})
    assert s.scheduler.api_min_interval["gemini"] == 4.0 and LLMResponse   # merged with the default, not replaced
