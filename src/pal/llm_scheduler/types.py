"""Shared types for the scheduler: who asked (task class), a handle on a submitted job (ticket), and per-turn statistics."""
from __future__ import annotations

import threading
import time
from enum import IntEnum
from typing import Any

from pal.errors import PalError


class TaskClass(IntEnum):
    """Who asked, which decides local-model priority. A lower number is served first."""

    AGENT = 0      # preparation for a request sent to an agent
    CHAT = 1       # preparation and easy answers for plain chat
    OTHER = 2      # background work nobody is waiting for


class SchedulerError(PalError):
    http_status = 503
    public = True


class QueueFullError(SchedulerError):
    """Too many jobs are already waiting."""


class ExpiredError(SchedulerError):
    """A background job waited too long and was dropped."""


class Ticket:
    """Returned by every submit. `result()` waits for the job; the other fields say how it went."""

    def __init__(self, label: str, task_class: TaskClass | None = None):
        self.label = label
        self.task_class = task_class
        self.submitted = time.monotonic()
        self.started: float | None = None        # first time the job began to run
        self.run_seconds = 0.0                   # time actually running (all slices added up)
        self.pauses = 0                          # times a higher-priority job paused it
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None

    @property
    def waited(self) -> float:
        """Seconds between submitting and first starting (what the user feels as queueing)."""
        return (self.started - self.submitted) if self.started is not None else time.monotonic() - self.submitted

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def set_result(self, value: Any) -> None:
        self._result = value
        self._done.set()

    def set_error(self, error: BaseException) -> None:
        self._error = error
        self._done.set()

    def result(self, timeout: float | None = None) -> Any:
        if not self._done.wait(timeout):
            raise TimeoutError(f"{self.label} did not finish within {timeout} s")
        if self._error is not None:
            raise self._error
        return self._result


class TurnStats:
    """Queueing numbers for one chat turn, filled in by the scheduler as the turn's jobs finish."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.wait_local = 0.0
        self.wait_api = 0.0
        self.local_pauses = 0

    def add_local(self, t: Ticket) -> None:
        with self._lock:
            self.wait_local += t.waited
            self.local_pauses += t.pauses

    def add_api(self, t: Ticket) -> None:
        with self._lock:
            self.wait_api += t.waited
