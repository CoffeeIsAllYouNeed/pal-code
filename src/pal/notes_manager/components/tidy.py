"""Tidy: merging duplicate and near-duplicate notes. No model is involved."""
from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager


def tidy_steps(memory, lock: AbstractContextManager, near: float = 0.9, batch: int = 100,
               on_removed: Callable[[str], None] | None = None) -> Generator[None, None, int]:
    """Merge duplicate and near-duplicate notes, keeping the one that was used most (then the newest). A generator that yields every
    `batch` notes, so a scheduler can pause it between batches; it returns how many notes were removed."""
    notes = sorted(memory.storage.read_all(), key=lambda n: (-n.retrieval_count, -n.timestamp))
    kept: list[set] = []
    removed = 0
    for i, n in enumerate(notes):
        words = set(" ".join(n.content.lower().split()).split())
        duplicate = any(len(words | k) and len(words & k) / len(words | k) >= near for k in kept) if len(words) >= 3 else False
        if duplicate:
            with lock:
                memory.manager.delete(n.id)
            removed += 1
            if on_removed:
                on_removed(n.id)
        else:
            kept.append(words)
        if (i + 1) % batch == 0:
            yield
    return removed
