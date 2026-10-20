import time
from dataclasses import dataclass

from pal.context_manager.utils.hashing import hash_segment
from pal.context_manager.utils.storage import InMemoryStorage, StorageBackend


@dataclass
class CacheEntry:
    content: str
    created_at: float
    hits: int = 0


class ContextCache:
    def __init__(self, storage: StorageBackend = None, ttl: float = 3600):
        self.storage = storage or InMemoryStorage()
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def get_or_store(self, content: str) -> tuple[bool, str]:
        key = hash_segment(content)
        entry = self.storage.get(key)

        if entry and (time.time() - entry.created_at) < self.ttl:
            entry.hits += 1
            self.storage.set(key, entry)
            self.hits += 1
            return True, key

        self.storage.set(key, CacheEntry(content=content, created_at=time.time()))
        self.misses += 1
        return False, key

    def resolve(self, key: str) -> str | None:
        entry = self.storage.get(key)
        return entry.content if entry else None

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size_bytes(self) -> int:
        return sum(len(self.storage.get(k).content.encode("utf-8")) for k in self.storage.keys())
