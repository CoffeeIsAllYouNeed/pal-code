"""Protection for the server: a rate limit, an authentication check and per-method timing."""
from __future__ import annotations

import hmac
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class RateLimiter:
    """A token bucket: it holds up to `per_minute` tokens, refills continuously, and each request takes one."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self._clock = clock
        self._last = clock()

    def allow(self) -> bool:
        """Take a token if one is available."""
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def check_token(provided: str | None, expected: str) -> bool:
    """True when no token is required, or the provided one matches (compared in constant time)."""
    if not expected:
        return True
    return provided is not None and hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


class Timing:
    """How many times each method was called and how long it took in total."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.total_ms: dict[str, float] = defaultdict(float)

    def record(self, method: str, seconds: float) -> None:
        self.calls[method] += 1
        self.total_ms[method] += seconds * 1000

    def summary(self) -> dict[str, Any]:
        return {m: {"calls": n, "avg_ms": round(self.total_ms[m] / n, 3)} for m, n in sorted(self.calls.items())}
