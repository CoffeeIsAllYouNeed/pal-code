"""
Storage Manager (AIOS section 3.6): persistent storage for agents --
files/knowledge bases they depend on, and durable storage for the
memory manager's RAM-to-disk swaps -- backed by local files
(LocalFileBackend) and a vector database (VectorStoreBackend, chromadb
when installed).

Two entry points into the same underlying storage, matching the two
callers section 3.6 describes:

- **Direct calls** (read/write/delete/list_keys/vector_upsert/
  vector_query) -- what the memory manager uses when it swaps a note to
  disk, scoped by the agent_id the memory manager passes in.
- **execute(StorageSyscall)**, reachable either directly or via the
  queue (submit/tick/drain) -- what an agent's own SDK calls
  (utils/sdk.py) go through: "the agent calls the storage API in the
  SDK, which is further converted into storage-related system calls
  and put into the storage queue... the storage manager then processes
  the requests in the queue."

Both paths converge on execute()/the same backends -- there's exactly
one storage manager per process, not a separate implementation per
caller.

When an AccessManagerPipeline (AIOS section 3.8) is supplied, execute()
authorizes every request before performing it: a cross-agent request
(syscall.requesting_agent_id differs from syscall.agent_id, the
resource owner) is checked against the privilege-group hashmap, and any
destructive operation -- DELETE, or a WRITE that overwrites an existing
key -- goes through user intervention regardless of who's making the
call. Without an access_manager, every request is permitted (the
pre-3.8 behavior), matching how the direct read/write/etc. calls above
still bypass access control entirely -- those are for the memory
manager's own swaps, which are not the cross-agent case 3.8 regulates.
"""
from pathlib import Path
from typing import Any, Optional

from pal.access_manager.components.base import OperationType
from pal.access_manager.pipeline import AccessManagerPipeline
from pal.paths import data_dir
from pal.storage_manager.components.base import StorageOp, StorageResponse, StorageSyscall
from pal.storage_manager.components.local_file_backend import LocalFileBackend
from pal.storage_manager.components.storage_queue import StorageSyscallQueue
from pal.storage_manager.components.vector_store_backend import VectorStoreBackend


class StorageManagerPipeline:
    def __init__(
        self,
        root: Optional[str] = None,
        vector_persist_directory: Optional[str] = None,
        access_manager: Optional[AccessManagerPipeline] = None,
    ):
        root_path = Path(root) if root else data_dir("storage")
        self.files = LocalFileBackend(root_path / "files")
        self.vectors = VectorStoreBackend(persist_directory=vector_persist_directory)
        self.queue = StorageSyscallQueue(self)
        self.access_manager = access_manager

    # -- direct calls (memory manager swaps, or any in-process caller) --------

    def read(self, agent_id: str, key: str) -> Optional[Any]:
        return self.files.read(agent_id, key)

    def write(self, agent_id: str, key: str, data: Any) -> None:
        self.files.write(agent_id, key, data)

    def delete(self, agent_id: str, key: str) -> None:
        self.files.delete(agent_id, key)

    def exists(self, agent_id: str, key: str) -> bool:
        return self.files.exists(agent_id, key)

    def list_keys(self, agent_id: str, prefix: str = "") -> list[str]:
        return self.files.list_keys(agent_id, prefix=prefix)

    def vector_upsert(self, agent_id: str, doc_id: str, text: str, metadata: dict = None) -> None:
        self.vectors.upsert(agent_id, doc_id, text, metadata or {})

    def vector_query(self, agent_id: str, query_text: str, top_k: int = 3) -> list[dict]:
        return self.vectors.query(agent_id, query_text, top_k=top_k)

    # -- system-call entry point (agent SDK calls, direct or via queue) -------

    def _operation_type(self, syscall: StorageSyscall) -> OperationType:
        if syscall.op == StorageOp.DELETE:
            return OperationType.DELETE
        if syscall.op in (StorageOp.READ, StorageOp.LIST, StorageOp.VECTOR_QUERY):
            return OperationType.READ
        if syscall.op == StorageOp.WRITE:
            # Overwriting an existing key destroys what was there --
            # treat it as destructive (user-intervention-gated) the same
            # as an explicit delete; a write to a fresh key is not.
            return OperationType.OVERWRITE if self.exists(syscall.agent_id, syscall.key) else OperationType.WRITE
        return OperationType.WRITE  # VECTOR_UPSERT

    def execute(self, syscall: StorageSyscall) -> StorageResponse:
        if self.access_manager is not None:
            requesting = syscall.requesting_agent_id or syscall.agent_id
            decision = self.access_manager.authorize(
                requesting_agent_id=requesting,
                target_agent_id=syscall.agent_id,
                operation=self._operation_type(syscall),
                resource=syscall.key or syscall.query_text or "",
            )
            if not decision.allowed:
                return StorageResponse(ok=False, error=f"access denied: {decision.reason}")

        try:
            if syscall.op == StorageOp.READ:
                return StorageResponse(ok=True, data=self.read(syscall.agent_id, syscall.key))
            if syscall.op == StorageOp.WRITE:
                self.write(syscall.agent_id, syscall.key, syscall.data)
                return StorageResponse(ok=True)
            if syscall.op == StorageOp.DELETE:
                self.delete(syscall.agent_id, syscall.key)
                return StorageResponse(ok=True)
            if syscall.op == StorageOp.LIST:
                return StorageResponse(ok=True, data=self.list_keys(syscall.agent_id, prefix=syscall.key or ""))
            if syscall.op == StorageOp.VECTOR_UPSERT:
                self.vector_upsert(syscall.agent_id, syscall.key, syscall.data, syscall.metadata)
                return StorageResponse(ok=True)
            if syscall.op == StorageOp.VECTOR_QUERY:
                return StorageResponse(
                    ok=True,
                    data=self.vector_query(syscall.agent_id, syscall.query_text, syscall.top_k),
                )
            return StorageResponse(ok=False, error=f"unknown storage op: {syscall.op}")
        except Exception as e:
            return StorageResponse(ok=False, error=str(e))

    # -- queue-driven path -- "put into the storage queue by the scheduler" --

    def submit(self, syscall: StorageSyscall) -> str:
        return self.queue.submit(syscall)

    def tick(self):
        return self.queue.process_next()

    def drain(self, max_iterations: int = 10_000):
        return self.queue.drain(max_iterations=max_iterations)

    def get_result(self, request_id: str):
        return self.queue.get_result(request_id)

    def forget(self, request_id: str) -> None:
        self.queue.forget(request_id)

    def purge_agent(self, agent_id: str) -> None:
        """Delete every file and vector stored for an agent (used when a chat session ends)."""
        self.files.purge(agent_id)
        self.vectors.delete_agent(agent_id)
