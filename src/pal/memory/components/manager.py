"""
Runtime memory manager for agent interaction history (AIOS section
3.5). Unlike a traditional OS memory manager, which manages physical
RAM pages, this one manages MemoryNote objects -- conversation turns,
tool-call results, and similar agent history -- between an in-process
RAM cache and on-disk persistence via StorageManager.

Every write is persisted to disk immediately: this project has no
concept of "unsaved" agent memory, durability isn't optional here, only
RAM residency is. The RAM cache exists purely as a bounded-size fast
path so repeated access to hot notes doesn't round-trip through disk
every time. When the cache's size crosses `high_water_mark` (default
80%) of `block_limit`, the manager evicts the coldest RAM-resident
notes by LRU-K backward-distance (components/lru_k.py) until it's back
under the mark. "Evicted" only means dropped from the RAM cache --
nothing is lost, since it was already durably written to disk on
admission; the next read() for that id reloads it from disk and
re-admits it to RAM (a swap-in).
"""
import time
from typing import Optional

from pal.memory.components.lru_k import LRUKTracker
from pal.memory.components.note import MemoryNote
from pal.memory.components.storage_manager import StorageManager


class MemoryManager:
    def __init__(
        self,
        storage: StorageManager,
        block_limit: int = 200,
        high_water_mark: float = 0.8,
        k: int = 2,
    ):
        self.storage = storage
        self.block_limit = block_limit
        self.high_water_mark = high_water_mark
        self.ram: dict[str, MemoryNote] = {}
        self.lru_k = LRUKTracker(k=k)

        # Warm the RAM cache from whatever's already durably persisted,
        # most-recently-touched first, up to the block limit -- so a
        # fresh process (a new session against an existing memory
        # store) doesn't start with a cold cache and immediately thrash
        # on its first few reads.
        for note in sorted(self.storage.read_all(), key=lambda n: n.last_accessed, reverse=True)[: self.block_limit]:
            self.ram[note.id] = note
            self.lru_k.record_access(note.id, note.last_accessed)

    def usage_ratio(self) -> float:
        return len(self.ram) / self.block_limit if self.block_limit else 0.0

    def write(self, note: MemoryNote) -> None:
        """Allocates (or updates) a note: persisted to disk immediately,
        cached in RAM, and recorded as an access. May trigger eviction
        if this pushes the RAM cache over its high-water mark."""
        self.storage.write(note)
        self.ram[note.id] = note
        self.lru_k.record_access(note.id)
        self._maybe_evict()

    def read(self, note_id: str) -> Optional[MemoryNote]:
        """Reads a note by id -- from RAM if resident, else swaps it in
        from disk. Returns None if the id doesn't exist anywhere."""
        note = self.ram.get(note_id)
        if note is None:
            note = self.storage.read(note_id)
            if note is None:
                return None
            self.ram[note_id] = note  # swap-in

        note.retrieval_count += 1
        note.last_accessed = time.time()
        self.lru_k.record_access(note_id, note.last_accessed)
        self.storage.write(note)  # persist the updated access metadata
        self._maybe_evict()
        return note

    def delete(self, note_id: str) -> None:
        self.ram.pop(note_id, None)
        self.storage.delete(note_id)
        self.lru_k.forget(note_id)

    def all_ids(self) -> list[str]:
        """Every note id known to the system, RAM-resident or swapped
        to disk."""
        return self.storage.list_ids()

    def _maybe_evict(self) -> None:
        while self.usage_ratio() >= self.high_water_mark and self.ram:
            victim_id = self.lru_k.eviction_order(list(self.ram.keys()))[0]
            # Already durable on disk from write() -- eviction here
            # only drops the RAM cache entry, nothing more needs to be
            # written. LRU-K access history is deliberately kept (not
            # forgotten) for the evicted item, so if it's swapped back
            # in later its recency history isn't reset to zero.
            del self.ram[victim_id]
