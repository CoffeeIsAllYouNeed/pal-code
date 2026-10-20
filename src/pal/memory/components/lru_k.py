"""
LRU-K tracking (O'Neil, O'Neil & Weikum, 1993) -- the eviction policy
AIOS section 3.5 specifies for its memory manager, used here to decide
which RAM-cached MemoryNotes get swapped to disk once the cache nears
capacity.

Plain LRU only looks at the single most recent access, so a note that
was accessed once, long ago, and a note accessed constantly until just
a moment ago can look identically "recently used" the instant both get
touched again -- LRU can't tell a one-off touch from a genuinely hot
item. LRU-K's backward K-distance (how long ago an item's K-th most
recent access happened, not just its most recent one) is what "at
least K times recently" in the spec actually cashes out to: an item
needs a real access HISTORY, not just a single recent touch, to count
as hot.
"""
import math
import time
from collections import deque


class LRUKTracker:
    def __init__(self, k: int = 2):
        self.k = k
        self._history: dict[str, deque] = {}

    def record_access(self, item_id: str, when: float = None) -> None:
        when = when if when is not None else time.time()
        hist = self._history.setdefault(item_id, deque(maxlen=self.k))
        hist.append(when)

    def backward_k_distance(self, item_id: str, now: float = None) -> float:
        """How long ago this item's K-th most recent access happened.
        Items with fewer than K recorded accesses ever get infinite
        distance -- they haven't earned "accessed at least K times
        recently" status yet, so they're always evicted ahead of
        anything that has."""
        now = now if now is not None else time.time()
        hist = self._history.get(item_id)
        if not hist or len(hist) < self.k:
            return math.inf
        return now - hist[0]  # oldest of the last K accesses

    def forget(self, item_id: str) -> None:
        self._history.pop(item_id, None)

    def eviction_order(self, candidate_ids: list[str], now: float = None) -> list[str]:
        """Returns candidate_ids ordered worst-to-best for retention --
        the FIRST id in the result is the one to evict first (largest
        backward K-distance = coldest). Python's sort is stable, so
        ties (most commonly: several items all still below K accesses,
        all at infinite distance) keep their relative order from
        candidate_ids -- in practice, whichever of them was inserted
        into the RAM cache earliest."""
        now = now if now is not None else time.time()
        return sorted(candidate_ids, key=lambda i: self.backward_k_distance(i, now), reverse=True)
