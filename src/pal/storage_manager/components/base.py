"""
Unified system-call layer for storage. Agents (via the SDK,
utils/sdk.py) and other modules (the memory manager, when it swaps
data to disk) never talk to a backend directly -- they issue a
StorageSyscall and get a StorageResponse back, same "syscall, not a
direct backend call" shape as llm_wrapper' LLMSyscall/LLMResponse.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StorageOp(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    LIST = "list"
    VECTOR_UPSERT = "vector_upsert"
    VECTOR_QUERY = "vector_query"


@dataclass
class StorageSyscall:
    """One storage request, always scoped to the agent that owns the
    data -- "the storage manager reads and writes data based on the
    agent ID passed from the memory manager" applies just as much to an
    agent's own direct SDK calls, so agent_id is required on every op,
    not just memory-manager swaps."""

    agent_id: str               # whose resource this operates on (the "owner"/target)
    op: StorageOp
    key: Optional[str] = None          # file ops: the key/filename; LIST: prefix filter
    data: Any = None                   # WRITE: value to store; VECTOR_UPSERT: document text
    query_text: Optional[str] = None   # VECTOR_QUERY
    top_k: int = 3                     # VECTOR_QUERY
    metadata: dict = field(default_factory=dict)  # VECTOR_UPSERT
    # Who is actually making this call -- distinct from agent_id when an
    # agent reads/writes ANOTHER agent's resource. None means "the owner
    # is calling on its own behalf", the common case, which needs no
    # access-manager cross-agent check (AIOS 3.8). Only meaningful when
    # a StorageManagerPipeline has an access_manager wired in; ignored
    # otherwise.
    requesting_agent_id: Optional[str] = None


@dataclass
class StorageResponse:
    ok: bool
    data: Any = None
    error: Optional[str] = None
