import threading
import time

import pytest

from pal.llm_scheduler import ExpiredError, LocalScheduler, QueueFullError, SchedulerError, TaskClass
from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper

A, C, BG = TaskClass.AGENT, TaskClass.CHAT, TaskClass.OTHER


@pytest.fixture
def sched():
    s = LocalScheduler()
    yield s
    s.close()


def block(s, cls=BG):
    """Occupy the worker so everything submitted next waits in the queues, in a known order."""
    started, release = threading.Event(), threading.Event()
    t = s.submit(lambda: (started.set(), release.wait(5)), cls, "blocker")
    assert started.wait(2)
    return release, t


def test_classes_are_served_agent_then_chat_then_other_and_first_come_first_served_inside_each(sched):
    release, blocker = block(sched)
    order = []
    tickets = [sched.submit(lambda n=n: order.append(n), cls, n)
               for n, cls in [("o1", BG), ("c1", C), ("a1", A), ("c2", C), ("o2", BG), ("a2", A)]]
    assert sched.pending() == {"agent": 2, "chat": 2, "other": 2}
    release.set()
    [t.result(5) for t in tickets]
    assert order == ["a1", "a2", "c1", "c2", "o1", "o2"]


def test_one_job_runs_at_a_time(sched):
    running, peak = [0], [0]
    lock = threading.Lock()

    def job():
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1

    [t.result(5) for t in [sched.submit(job, C) for _ in range(8)]]
    assert peak[0] == 1


def test_atomic_jobs_are_never_paused_an_agent_waits_for_them_to_finish(sched):
    release, blocker = block(sched, C)
    log = []
    agent = sched.submit(lambda: log.append("agent"), A)
    time.sleep(0.05)
    assert log == [] and blocker.pauses == 0          # the agent is waiting; the running chat job is not interrupted
    log.append("chat done")
    release.set()
    agent.result(5)
    assert log == ["chat done", "agent"] and blocker.pauses == 0


def test_a_steps_job_is_paused_between_pieces_for_a_better_job_then_resumes(sched):
    log = []
    agent = []

    def steps():
        for i in range(1, 5):
            log.append(f"piece{i}")
            if i == 2:
                agent.append(sched.submit(lambda: log.append("agent"), A, "agent"))
            yield
        return "done"

    t = sched.submit_steps(steps(), BG, "summary")
    assert t.result(5) == "done"
    agent[0].result(5)
    assert log == ["piece1", "piece2", "agent", "piece3", "piece4"] and t.pauses == 1


def test_chat_pauses_other_work_and_agent_pauses_chat_work(sched):
    log, tickets = [], {}

    def other():
        for i in (1, 2, 3):
            log.append(f"o{i}")
            if i == 1:
                tickets["chat"] = sched.submit_steps(chat(), C, "chat")
            yield

    def chat():
        for i in (1, 2):
            log.append(f"c{i}")
            if i == 1:
                tickets["agent"] = sched.submit(lambda: log.append("a"), A, "agent")
            yield

    t = sched.submit_steps(other(), BG, "other")
    t.result(5)
    tickets["chat"].result(5)
    assert log == ["o1", "c1", "a", "c2", "o2", "o3"]
    assert (t.pauses, tickets["chat"].pauses) == (1, 1)


def test_a_paused_job_resumes_before_later_arrivals_of_its_own_class(sched):
    log = []

    def steps(name, hook=None):
        for i in (1, 2):
            log.append(f"{name}{i}")
            if hook and i == 1:
                hook()
            yield

    later = []
    first = sched.submit_steps(steps("A", lambda: (sched.submit(lambda: log.append("agent"), A),
                                                   later.append(sched.submit_steps(steps("B"), BG)))), BG)
    first.result(5)
    later[0].result(5)
    assert log == ["A1", "agent", "A2", "B1", "B2"]


def test_a_job_is_paused_at_most_max_pauses_times_then_runs_to_the_end():
    s = LocalScheduler(max_pauses=1)
    log, agents = [], []

    def steps():
        for i in range(1, 5):
            log.append(f"p{i}")
            if i == 1:
                agents.append(s.submit(lambda: log.append("agent1"), A))
            if i == 3:
                agents.append(s.submit(lambda: log.append("agent2"), A))
            yield

    t = s.submit_steps(steps(), BG)
    t.result(5)
    [a.result(5) for a in agents]
    assert log == ["p1", "agent1", "p2", "p3", "p4", "agent2"] and t.pauses == 1
    s.close()


class StreamCore(LLMWrapper):
    name = "stream"

    def __init__(self, sched, log):
        self.sched, self.log, self.calls, self.closed, self.fired = sched, log, [], 0, False

    def chat(self, syscall):
        raise AssertionError("a stream job must not call chat")

    def token_count(self, text):
        return 1

    def capabilities(self):
        return None

    def stream_chat(self, syscall):
        self.calls.append(syscall.messages)
        self.log.append(f"stream{len(self.calls)}")
        try:
            resumed = syscall.messages[-1]["content"].startswith("[Continue")
            for chunk in (["three ", "four "] if resumed else ["one ", "two ", "three ", "four "]):
                if chunk == "two " and not self.fired:
                    self.fired = True
                    self.sched.submit(lambda: self.log.append("agent"), A, "agent")
                yield chunk
        finally:
            self.closed += 1


def test_a_text_stream_is_closed_for_a_better_job_and_resumed_by_continuing_from_its_text(sched):
    log = []
    core = StreamCore(sched, log)
    t = sched.submit_stream(core, LLMSyscall(messages=[{"role": "user", "content": "count"}]), C, "answer")
    assert t.result(5) == "one two three four "
    assert log == ["stream1", "agent", "stream2"] and t.pauses == 1
    assert core.closed == 2                                              # the first stream was closed: the model stopped
    cont = core.calls[1]
    assert cont[-2] == {"role": "assistant", "content": "one two "} and cont[-1]["content"].startswith("[Continue")
    assert cont[0] == {"role": "user", "content": "count"}


def test_a_stream_nobody_waits_for_is_never_paused(sched):
    log = []
    core = StreamCore(sched, log)
    core.fired = True
    t = sched.submit_stream(core, LLMSyscall(messages=[{"role": "user", "content": "x"}]), BG)
    assert t.result(5) == "one two three four " and t.pauses == 0 and len(core.calls) == 1


def test_background_jobs_that_wait_too_long_are_dropped(sched):
    release, _ = block(sched, C)
    stale = sched.submit_steps(iter(()), BG, "stale", expires_after=0.05)
    fresh = sched.submit_steps(iter(()), BG, "fresh", expires_after=30)
    time.sleep(0.15)
    release.set()
    with pytest.raises(ExpiredError, match="waited too long"):
        stale.result(5)
    fresh.result(5)
    s = LocalScheduler(other_stale_seconds=0.05)
    r2, _ = block(s, C)
    t = s.submit_steps(iter(()), BG)
    time.sleep(0.15)
    r2.set()
    with pytest.raises(ExpiredError):
        t.result(5)
    s.close()


def test_only_background_work_expires_never_agent_or_chat_work(sched):
    release, _ = block(sched)
    t = sched.submit(lambda: "ok", A)
    time.sleep(0.05)
    release.set()
    assert t.result(5) == "ok"


def test_a_full_queue_refuses_new_work():
    s = LocalScheduler(max_queue=2)
    release, _ = block(s)
    s.submit(lambda: 1)
    s.submit(lambda: 2)
    with pytest.raises(QueueFullError, match="too many"):
        s.submit(lambda: 3)
    release.set()
    s.close()


def test_errors_reach_the_submitter_and_the_worker_keeps_going(sched):
    bad = sched.submit(lambda: 1 / 0, C, "bad")
    good = sched.submit(lambda: "fine", C, "good")
    with pytest.raises(ZeroDivisionError):
        bad.result(5)
    assert good.result(5) == "fine"
    broken = sched.submit_steps((x for x in [1] if (_ for _ in ()).throw(RuntimeError("step failed"))), BG)
    with pytest.raises(RuntimeError, match="step failed"):
        broken.result(5)
    assert sched.submit(lambda: 2).result(5) == 2


def test_tickets_report_how_long_a_job_waited_and_ran(sched):
    release, _ = block(sched)
    t = sched.submit(lambda: time.sleep(0.05), C)
    time.sleep(0.1)
    release.set()
    t.result(5)
    assert t.waited >= 0.09 and 0.04 <= t.run_seconds < 0.5 and t.pauses == 0
    assert sched.history[-1]["class"] == "CHAT" and sched.history[-1]["pauses"] == 0
    with pytest.raises(TimeoutError):
        sched.submit(lambda: time.sleep(0.3)).result(0.01)


def test_closing_stops_the_worker_and_fails_waiting_jobs():
    s = LocalScheduler()
    release, _ = block(s)
    waiting = s.submit(lambda: 1)
    s.close()
    with pytest.raises(SchedulerError, match="closed"):
        waiting.result(5)
    with pytest.raises(SchedulerError, match="closed"):
        s.submit(lambda: 2)
    release.set()
