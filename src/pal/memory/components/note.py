"""
A single persistent memory unit. Design adapted from AIOS
(github.com/agiresearch/AIOS aios/memory/note.py) but trimmed to what
Pal actually needs -- no evolution_history/links graph, since that
requires the heavier retrieval stack (chromadb/qdrant) AIOS uses, which
this lightweight module deliberately avoids.
"""
import time
import uuid
from typing import Optional


class MemoryNote:
    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        last_accessed: Optional[float] = None,
        retrieval_count: int = 0,
        tags: Optional[list[str]] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.session_id = session_id
        now = time.time()
        self.timestamp = timestamp if timestamp is not None else now
        self.last_accessed = last_accessed if last_accessed is not None else now
        self.retrieval_count = retrieval_count
        self.tags = tags or []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "retrieval_count": self.retrieval_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNote":
        return cls(
            content=d["content"],
            id=d.get("id"),
            session_id=d.get("session_id"),
            timestamp=d.get("timestamp"),
            last_accessed=d.get("last_accessed"),
            retrieval_count=d.get("retrieval_count", 0),
            tags=d.get("tags", []),
        )
