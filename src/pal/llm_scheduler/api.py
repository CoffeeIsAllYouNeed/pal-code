"""The API model's scheduler: first come, first served, one call at a time, per API key.

With one key there is one line, and chat, the coding agent and the documentation agent all wait in it, strictly in the order
they arrive. Two things protect a free tier: calls start at least `interval` seconds apart, and a rate-limit error (429,
quota) pauses the line and retries the same call after a growing delay instead of failing everyone behind it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pal.llm_scheduler.types import QueueFullError, SchedulerError, Ticket

log = logging.getLogger(__name__)


def is_rate_limit(error: BaseException) -> bool:
    for attr in ("status_code", "code"):
        if getattr(error, attr, None) == 429:
            return True
    resp = getattr(error, "response", None)
    if getattr(resp, "status_code", None) == 429:
        return True
    low = str(error).lower()
    return any(k in low for k in ("429", "resource_exhausted", "rate limit", "quota"))


class _ApiJob:
    def __init__(self, ticket: Ticket, fn: Callable[[], Any]):
        self.ticket, self.fn = ticket, fn


class ApiLine:
    def __init__(self, interval: float, retries: int, backoff: float, max_queue: int):
        self.interval, self.retries, self.backoff, self.max_queue = interval, retries, backoff, max_queue
        self._queue: deque[_ApiJob] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="pal-api-line", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any], label: str = "api call") -> Ticket:
        t = Ticket(label)
        with self._cond:
            if self._closed:
                raise SchedulerError("the scheduler is closed")
            if len(self._queue) >= self.max_queue:
                raise QueueFullError(f"too many API calls are waiting ({self.max_queue})")
            self._queue.append(_ApiJob(t, fn))
            self._cond.notify()
        return t

    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    def _worker(self) -> None:
        last_start = 0.0
        while True:
            with self._cond:
                while not self._queue:
                    if self._closed:
                        return
                    self._cond.wait()
                job = self._queue.popleft()
            gap = self.interval - (time.monotonic() - last_start)
            if gap > 0:
                time.sleep(gap)
            t = job.ticket
            t.started = last_start = time.monotonic()
            attempt = 0
            while True:
                began = time.monotonic()
                try:
                    value = job.fn()
                except BaseException as e:  # noqa: BLE001  (the caller gets the original error)
                    t.run_seconds += time.monotonic() - began
                    if is_rate_limit(e) and attempt < self.retries and not self._closed:
                        delay = self.backoff * (2 ** attempt)
                        attempt += 1
                        log.warning("rate limit on the API line: waiting %.1f s, then retry %d of %d", delay, attempt, self.retries)
                        time.sleep(delay)
                        last_start = time.monotonic()
                        continue
                    t.set_error(e)
                    break
                t.run_seconds += time.monotonic() - began
                t.set_result(value)
                break

    def close(self) -> None:
        with self._cond:
            self._closed = True
            dropped = list(self._queue)
            self._queue.clear()
            self._cond.notify_all()
        for j in dropped:
            j.ticket.set_error(SchedulerError("the scheduler was closed"))


class ApiScheduler:
    """One ApiLine per API key (the key is part of the line name), created on first use."""

    def __init__(self, retries: int = 2, backoff: float = 5.0, max_queue: int = 100, enabled: bool = True):
        self.retries, self.backoff, self.max_queue, self.enabled = retries, backoff, max_queue, enabled
        self._lines: dict[str, ApiLine] = {}
        self._lock = threading.Lock()

    def submit(self, line_key: str, fn: Callable[[], Any], interval: float = 0.0, label: str = "api call") -> Ticket:
        if not self.enabled:                       # scheduling off: run right here, no line
            t = Ticket(label)
            t.started = time.monotonic()
            try:
                t.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                t.set_error(e)
            return t
        with self._lock:
            line = self._lines.get(line_key)
            if line is None:
                line = self._lines[line_key] = ApiLine(interval, self.retries, self.backoff, self.max_queue)
            line.interval = interval
        return line.submit(fn, label)

    def pending(self) -> dict[str, int]:
        with self._lock:
            return {k: ln.pending() for k, ln in self._lines.items()}

    def close(self) -> None:
        with self._lock:
            lines, self._lines = list(self._lines.values()), {}
        for ln in lines:
            ln.close()
