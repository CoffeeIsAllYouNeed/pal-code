"""The local model's scheduler: preemptive priority, first come first served inside each class.

One job runs at a time. Classes in order: AGENT, then CHAT, then OTHER. When a job of a better class is waiting, the
running job is paused at its next safe point and the better job goes first; the paused job then resumes before any later
arrival of its own class, so first come first served is kept inside a class.

Three kinds of job, because a model call cannot always be paused:
  * atomic     one call that finishes (route, trim, verify: structured and short). Never paused.
  * steps      a generator of small pieces (summarise a section, tidy a note). Paused between pieces: no state to save.
  * stream     a plain-text generation. Paused between chunks: the stream is closed (so the model stops), the text so far is
               kept, and the job resumes by continuing from that text. Only for cores that really stream.
A job can be paused at most `max_pauses` times, then it runs to the end, so it cannot be postponed forever.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from pal.llm_scheduler.types import ExpiredError, QueueFullError, SchedulerError, TaskClass, Ticket
from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper

log = logging.getLogger(__name__)


class _Job:
    preemptible = False

    def __init__(self, ticket: Ticket, cls: TaskClass, deadline: float | None = None):
        self.ticket, self.cls, self.deadline = ticket, cls, deadline
        self.result: Any = None

    def run(self, should_stop: Callable[[], bool]) -> bool:  # True when finished
        raise NotImplementedError


class _AtomicJob(_Job):
    def __init__(self, ticket: Ticket, cls: TaskClass, fn: Callable[[], Any], deadline: float | None = None):
        super().__init__(ticket, cls, deadline)
        self.fn = fn

    def run(self, should_stop: Callable[[], bool]) -> bool:
        self.result = self.fn()
        return True


class _StepsJob(_Job):
    preemptible = True

    def __init__(self, ticket: Ticket, cls: TaskClass, steps: Iterator[Any], deadline: float | None = None):
        super().__init__(ticket, cls, deadline)
        self.steps = steps

    def run(self, should_stop: Callable[[], bool]) -> bool:
        while True:
            try:
                next(self.steps)
            except StopIteration as done:
                self.result = done.value
                return True
            if should_stop():
                return False


class _StreamJob(_Job):
    preemptible = True

    def __init__(self, ticket: Ticket, cls: TaskClass, core: LLMWrapper, syscall: LLMSyscall):
        super().__init__(ticket, cls)
        self.core, self.syscall, self.text = core, syscall, ""

    def _continuation(self) -> LLMSyscall:
        from pal.orchestrator.components.context_interrupt import (  # lazy: avoids an import cycle
            ContextInterruptController,
            ContextSnapshot,
            InterruptResult,
        )
        partial = InterruptResult(complete=False, text_so_far=self.text, elapsed=0.0,
                                  snapshot=ContextSnapshot(mode="text", text_so_far=self.text))
        return ContextInterruptController().build_continuation(self.syscall, partial)

    def run(self, should_stop: Callable[[], bool]) -> bool:
        stream = iter(self.core.stream_chat(self.syscall if not self.text else self._continuation()))
        try:
            chunk = next(stream, None)
            while chunk is not None:
                self.text += chunk
                chunk = next(stream, None)                  # look at the next chunk before deciding to stop,
                if chunk is None:                           # so a stream that has just ended is never paused
                    break
                if should_stop():
                    self.text += chunk
                    return False
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()                                     # closing the stream makes the model stop generating
        self.result = self.text
        return True


class LocalScheduler:
    def __init__(self, max_pauses: int = 2, other_stale_seconds: float = 600.0, max_queue: int = 200):
        self.max_pauses = max_pauses
        self.other_stale_seconds = other_stale_seconds
        self.max_queue = max_queue
        self._queues: dict[TaskClass, deque[_Job]] = {c: deque() for c in TaskClass}
        self._cond = threading.Condition()
        self._closed = False
        self.history: deque[dict] = deque(maxlen=500)
        self._thread = threading.Thread(target=self._worker, name="pal-local-scheduler", daemon=True)
        self._thread.start()

    # -- submitting -------------------------------------------------------------------------------------------
    def submit(self, fn: Callable[[], Any], task_class: TaskClass = TaskClass.CHAT, label: str = "local job") -> Ticket:
        """An atomic job: runs to the end, never paused."""
        t = Ticket(label, task_class)
        return self._enqueue(_AtomicJob(t, task_class, fn, self._deadline(task_class, None)))

    def submit_steps(self, steps: Iterator[Any], task_class: TaskClass = TaskClass.OTHER, label: str = "local steps",
                     expires_after: float | None = None) -> Ticket:
        """A job made of small pieces: `steps` yields after each piece; its return value is the result."""
        t = Ticket(label, task_class)
        return self._enqueue(_StepsJob(t, task_class, steps, self._deadline(task_class, expires_after)))

    def submit_stream(self, core: LLMWrapper, syscall: LLMSyscall, task_class: TaskClass = TaskClass.CHAT,
                      label: str = "local answer") -> Ticket:
        """A plain-text generation that can be paused and resumed. The result is the whole text."""
        t = Ticket(label, task_class)
        return self._enqueue(_StreamJob(t, task_class, core, syscall))

    def _deadline(self, cls: TaskClass, expires_after: float | None) -> float | None:
        if expires_after is None and cls == TaskClass.OTHER:
            expires_after = self.other_stale_seconds
        return time.monotonic() + expires_after if expires_after else None

    def _enqueue(self, job: _Job) -> Ticket:
        with self._cond:
            if self._closed:
                raise SchedulerError("the scheduler is closed")
            if sum(len(q) for q in self._queues.values()) >= self.max_queue:
                raise QueueFullError(f"too many local jobs are waiting ({self.max_queue})")
            self._queues[job.cls].append(job)
            self._cond.notify()
        return job.ticket

    def pending(self) -> dict[str, int]:
        with self._cond:
            return {c.name.lower(): len(q) for c, q in self._queues.items()}

    # -- the worker ----------------------------------------------------------------------------------------------
    def _higher_waiting(self, cls: TaskClass) -> bool:
        with self._cond:
            return any(self._queues[c] for c in TaskClass if c < cls)

    def _next_locked(self) -> _Job | None:
        for cls in TaskClass:
            q = self._queues[cls]
            while q:
                job = q.popleft()
                if job.deadline is not None and time.monotonic() > job.deadline and job.ticket.started is None:
                    job.ticket.set_error(ExpiredError(f"{job.ticket.label} waited too long and was dropped"))
                    continue
                return job
        return None

    def _worker(self) -> None:
        while True:
            with self._cond:
                job = self._next_locked()
                while job is None:
                    if self._closed:
                        return
                    self._cond.wait(timeout=1.0)
                    job = self._next_locked()
            self._run(job)

    def _run(self, job: _Job) -> None:
        t = job.ticket
        if t.started is None:
            t.started = time.monotonic()
        began = time.monotonic()

        def should_stop() -> bool:
            return job.preemptible and t.pauses < self.max_pauses and self._higher_waiting(job.cls)

        try:
            finished = job.run(should_stop)
        except BaseException as e:  # noqa: BLE001  (the error belongs to whoever submitted the job)
            t.run_seconds += time.monotonic() - began
            t.set_error(e)
            return
        t.run_seconds += time.monotonic() - began
        if finished:
            self.history.append({"label": t.label, "class": job.cls.name, "waited": round(t.waited, 4),
                                 "ran": round(t.run_seconds, 4), "pauses": t.pauses})
            t.set_result(job.result)
            return
        t.pauses += 1
        log.info("paused %s (%s) for a higher-priority job; pause %d", t.label, job.cls.name, t.pauses)
        with self._cond:
            self._queues[job.cls].appendleft(job)      # resumes before later arrivals of its own class

    def close(self) -> None:
        with self._cond:
            self._closed = True
            dropped = [j for q in self._queues.values() for j in q]
            for q in self._queues.values():
                q.clear()
            self._cond.notify_all()
        for j in dropped:
            j.ticket.set_error(SchedulerError("the scheduler was closed"))
