"""Code work (converting files, building the code graph) runs in code, beside the LLM lines. It does not wait for the model,
but it uses the CPU the local model also needs, so only a few run at once."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class CodeSlots:
    def __init__(self, workers: int = 2):
        self._sem = threading.BoundedSemaphore(workers) if workers > 0 else None
        self._lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def run(self, fn: Callable[[], Any]) -> Any:
        if self._sem is None:
            return fn()
        with self._sem:
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            try:
                return fn()
            finally:
                with self._lock:
                    self.running -= 1
