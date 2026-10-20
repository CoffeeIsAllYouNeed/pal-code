"""
Adapter presenting the same narrow interface
pal.memory.components.storage_manager.StorageManager exposes
(write/read/delete/exists/list_ids/read_all, all keyed by MemoryNote.id)
but backed by the general-purpose StorageManagerPipeline, scoped to one
agent_id.

This is the actual integration point for "when the agent's memory
usage exceeds the allocated limit, the memory manager calls the storage
manager to swap the data into the disk... the storage manager reads and
writes data based on the agent ID passed from the memory manager":
MemoryManager doesn't know or care which storage implementation sits
behind this interface, only that it has one -- so it works identically
whether it's given this adapter (backed by the shared, agent-scoped
StorageManagerPipeline) or the standalone pal.memory.components.
storage_manager.StorageManager (a private, unscoped JSON-per-note
store, still the default when no agent_id/storage_manager is supplied).
"""
from typing import Optional

from pal.memory.components.note import MemoryNote
from pal.storage_manager.pipeline import StorageManagerPipeline

_MEMORY_PREFIX = "memory/"


class AgentScopedMemoryStorage:
    def __init__(self, storage_manager: StorageManagerPipeline, agent_id: str):
        self.storage_manager = storage_manager
        self.agent_id = agent_id

    def _key(self, note_id: str) -> str:
        return f"{_MEMORY_PREFIX}{note_id}.json"

    def write(self, note: MemoryNote) -> None:
        self.storage_manager.write(self.agent_id, self._key(note.id), note.to_dict())

    def read(self, note_id: str) -> Optional[MemoryNote]:
        data = self.storage_manager.read(self.agent_id, self._key(note_id))
        if data is None:
            return None
        return MemoryNote.from_dict(data)

    def delete(self, note_id: str) -> None:
        self.storage_manager.delete(self.agent_id, self._key(note_id))

    def exists(self, note_id: str) -> bool:
        return self.storage_manager.exists(self.agent_id, self._key(note_id))

    def list_ids(self) -> list[str]:
        keys = self.storage_manager.list_keys(self.agent_id, prefix=_MEMORY_PREFIX)
        return [k[len(_MEMORY_PREFIX):-len(".json")] for k in keys if k.endswith(".json")]

    def read_all(self) -> list[MemoryNote]:
        notes = []
        for note_id in self.list_ids():
            note = self.read(note_id)
            if note is not None:
                notes.append(note)
        return notes
