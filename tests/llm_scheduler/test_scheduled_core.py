import threading
import time

import pytest

from pal.llm_scheduler import (
    LLMScheduler,
    LocalScheduler,
    ScheduledWrapper,
    TaskClass,
    current_stats,
    task,
    track,
)
from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper

A, C, BG = TaskClass.AGENT, TaskClass.CHAT, TaskClass.OTHER
MSG = [{"role": "user", "content": "hello there"}]


class PlainCore(LLMWrapper):
    """chat() only: stream_chat is the default (whole answer as one chunk)."""
    name = "plain"

    def __init__(self, log=None):
        self.config = {"x": 1}
        self.log = log if log is not None else []

    def chat(self, syscall):
        self.log.append(syscall.messages[-1]["content"])
        return LLMResponse(text="reply", prompt_tokens=7, completion_tokens=2)

    def token_count(self, text):
        return 3

    def capabilities(self):
        return "caps"

    def special(self):
        return "special"


class StreamingCore(PlainCore):
    def stream_chat(self, syscall):
        yield from ["str", "eam"]


@pytest.fixture
def sched():
    s = LocalScheduler()
    yield s
    s.close()


def test_without_a_scheduler_the_core_is_a_plain_passthrough():
    core = ScheduledWrapper(PlainCore(), None)
    assert core.chat(LLMSyscall(messages=MSG)).text == "reply" and core.token_count("x") == 3 and core.capabilities() == "caps"
    assert core.special() == "special" and core.config == {"x": 1}


def test_calls_become_jobs_in_the_class_of_the_running_turn_so_agents_go_first(sched):
    log = []
    core = ScheduledWrapper(PlainCore(log), sched)
    started, release = threading.Event(), threading.Event()
    sched.submit(lambda: (started.set(), release.wait(5)), BG, "blocker")
    started.wait(2)

    def caller(cls, text):
        with task(cls):
            core.chat(LLMSyscall(messages=[{"role": "user", "content": text}]))

    chat = threading.Thread(target=caller, args=(C, "chat call"))
    chat.start()
    while sched.pending()["chat"] < 1:
        time.sleep(0.01)
    agent = threading.Thread(target=caller, args=(A, "agent call"))
    agent.start()
    while sched.pending()["agent"] < 1:
        time.sleep(0.01)
    release.set()
    chat.join(5)
    agent.join(5)
    assert log == ["agent call", "chat call"]


def test_the_default_class_is_chat():
    s = LocalScheduler()
    core = ScheduledWrapper(PlainCore(), s)
    core.chat(LLMSyscall(messages=MSG))
    assert s.history[-1]["class"] == "CHAT"
    s.close()


def test_a_preemptible_call_on_a_real_streaming_core_runs_as_a_pausable_stream(sched):
    core = ScheduledWrapper(StreamingCore(), sched)
    with task(C, preemptible=True):
        r = core.chat(LLMSyscall(messages=MSG))
    assert r.text == "stream" and r.meta["streamed"] is True and r.completion_tokens >= 1 and r.prompt_tokens >= 1


def test_a_core_without_real_streaming_keeps_the_atomic_path_and_its_exact_token_counts(sched):
    core = ScheduledWrapper(PlainCore(), sched)
    with task(C, preemptible=True):
        r = core.chat(LLMSyscall(messages=MSG))
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("reply", 7, 2) and "streamed" not in r.meta


def test_waits_and_pauses_are_added_to_the_turns_stats(sched):
    core = ScheduledWrapper(PlainCore(), sched)
    started, release = threading.Event(), threading.Event()
    sched.submit(lambda: (started.set(), release.wait(5)), BG)
    started.wait(2)
    result = {}

    def turn():
        with track() as stats, task(C):
            core.chat(LLMSyscall(messages=MSG))
            result["stats"] = stats
            result["inside"] = current_stats() is stats

    t = threading.Thread(target=turn)
    t.start()
    time.sleep(0.12)
    release.set()
    t.join(5)
    assert result["stats"].wait_local >= 0.1 and result["stats"].local_pauses == 0 and result["inside"]
    assert current_stats() is None


def test_errors_from_the_model_reach_the_caller(sched):
    class Boom(PlainCore):
        def chat(self, syscall):
            raise ConnectionError("ollama down")

    with pytest.raises(ConnectionError, match="ollama down"):
        ScheduledWrapper(Boom(), sched).chat(LLMSyscall(messages=MSG))


def test_the_scheduler_facade_wraps_once_reports_a_snapshot_and_can_be_switched_off():
    s = LLMScheduler({"api_min_interval": {"gemini": 4.0, "openai": 0.5}})
    core = PlainCore()
    wrapped = s.wrap_local(core)
    assert isinstance(wrapped, ScheduledWrapper) and s.wrap_local(wrapped) is wrapped and s.wrap_local(None) is None
    assert s.interval_for("gemini") == 4.0 and s.interval_for("openai") == 0.5 and s.interval_for("anthropic") == 0.0
    assert s.snapshot() == {"enabled": True, "local_waiting": {"agent": 0, "chat": 0, "other": 0}, "api_waiting": {}}
    s.close()
    off = LLMScheduler({"enabled": False})
    assert off.local is None and off.wrap_local(core).chat(LLMSyscall(messages=MSG)).text == "reply"
    assert off.snapshot()["local_waiting"] == {}
    off.close()
