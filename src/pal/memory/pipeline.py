"""
Persistent, cross-session agent memory (AIOS section 3.5) -- distinct
from context_manager's ContextPager, which only pages within ONE open
conversation and forgets everything when the process exits.

Composes MemoryManager (RAM cache + block-limit-triggered LRU-K
eviction -- see components/manager.py, components/lru_k.py) with
lightweight relevance scoring for retrieve(). Disk persistence for the
manager's swaps is provided either by a private, unscoped local store
(components/storage_manager.py, the default) or -- when this Pipeline
is given an agent_id and a shared StorageManagerPipeline -- by the
general-purpose, agent-scoped Storage Manager (AIOS section 3.6, see
pal/storage_manager/), matching "the memory manager calls the
storage manager to swap the data into the disk... based on the agent ID
passed from the memory manager."
"""
import time
from pathlib import Path
from typing import Optional

from pal.context_manager.utils.relevance import embedding_similarity, recency_score
from pal.memory.components.manager import MemoryManager
from pal.memory.components.note import MemoryNote
from pal.memory.components.storage_manager import StorageManager
from pal.memory.utils.keyword_overlap import keyword_overlap

_DEFAULT_STORE_ROOT = Path.home() / ".pal" / "memory"


class MemoryPipeline:
    """
    add(content) -> writes a new note through the MemoryManager (RAM
      cache + durable disk persistence, LRU-K eviction once the RAM
      cache nears block_limit).
    retrieve(query, top_k) -> scores every persisted note (RAM- or
      disk-resident) by embedding similarity (or keyword overlap if
      embeddings unavailable) blended with recency. Scoring has to
      inspect every note's content regardless of tier -- this is a
      lightweight, non-indexed store, there's no cheaper way to rank
      relevance -- but only the notes that make the returned top_k are
      actually READ through the manager, which is what counts as an
      access for LRU-K and may swap a disk-resident note back into RAM.
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        block_limit: int = 200,
        high_water_mark: float = 0.8,
        lru_k: int = 2,
        agent_id: Optional[str] = None,
        storage_manager=None,
    ):
        if storage_manager is not None:
            # Central, agent-scoped Storage Manager (AIOS 3.6) -- the
            # memory manager's swaps go through the same backend an
            # agent's own SDK calls would use, scoped by agent_id.
            from pal.storage_manager.utils.memory_adapter import AgentScopedMemoryStorage
            self.storage = AgentScopedMemoryStorage(storage_manager, agent_id or "default")
        else:
            # Standalone fallback -- a private, unscoped local JSON-
            # per-note store. Used when this Pipeline isn't wired into
            # a broader orchestrator/StorageManagerPipeline (e.g. the
            # standalone webapp).
            root = Path(store_path) if store_path else _DEFAULT_STORE_ROOT
            self.storage = StorageManager(root)

        self.manager = MemoryManager(self.storage, block_limit=block_limit, high_water_mark=high_water_mark, k=lru_k)

    def add(self, content: str, session_id: Optional[str] = None, tags: Optional[list[str]] = None) -> MemoryNote:
        note = MemoryNote(content=content, session_id=session_id, tags=tags)
        self.manager.write(note)
        return note

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[MemoryNote]:
        all_notes = self.storage.read_all()
        if not all_notes:
            return []

        contents = [n.content for n in all_notes]
        sims = embedding_similarity(query, contents)

        # embedding_similarity returns all-zero if sentence-transformers
        # isn't installed -- fall back to keyword overlap in that case.
        if all(s == 0.0 for s in sims):
            sims = [keyword_overlap(query, c) for c in contents]

        now = time.time()
        scored = []
        for note, sim in zip(all_notes, sims):
            rec = recency_score(note.timestamp, now)
            # 70% relevance, 30% recency -- relevance should dominate for
            # memory retrieval (unlike hot/cold conversation paging,
            # where recency matters more).
            score = 0.7 * sim + 0.3 * rec
            scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [note.id for score, note in scored if score >= min_score][:top_k]

        # read() (not the scored `note` objects directly) so this
        # registers as a real access for LRU-K and can swap a
        # disk-only note into RAM.
        results = [self.manager.read(nid) for nid in top_ids]
        return [r for r in results if r is not None]

    def clear(self) -> None:
        for note_id in list(self.storage.list_ids()):
            self.manager.delete(note_id)

    def stats(self) -> dict:
        """RAM/disk split and cache pressure -- useful for observing the
        LRU-K eviction behavior directly rather than just its effects."""
        return {
            "ram_count": len(self.manager.ram),
            "disk_count": len(self.storage.list_ids()),
            "usage_ratio": round(self.manager.usage_ratio(), 3),
            "block_limit": self.manager.block_limit,
        }

    def __len__(self) -> int:
        return len(self.storage.list_ids())
