"""
Storage SDK: what an agent calls to read/write its own persistent data
during runtime -- distinct from the memory manager's own direct calls
into the same storage manager (see utils/memory_adapter.py).

"The agent calls the storage API in the SDK, which is further
converted into storage-related system calls and put into the storage
queue by the scheduler. The storage manager then processes the
requests in the queue." read()/write()/etc. here build a StorageSyscall
and submit it to the shared StorageManagerPipeline's queue, then drain
that queue until this specific request completes -- so the call still
looks synchronous to the agent even though it went through the same
centralized queue every other storage request goes through, the same
way Pipeline._dispatch_chat drains the LLM syscall scheduler for its
own request in orchestrator/components/syscall_scheduler.py.

read_from()/write_from()/delete_from() are the cross-agent path AIOS
3.8's Access Manager regulates: they set requesting_agent_id to this
SDK's own agent_id while targeting a DIFFERENT agent's resource, so if
the StorageManagerPipeline has an access_manager wired in, that request
gets checked against the privilege-group hashmap (and, for delete/
overwrite, gated behind user intervention) before it's allowed to
proceed. Plain read()/write()/delete() never trigger a cross-agent
check -- they always act on this SDK's own agent_id as both requester
and target.
"""
from typing import Any, Optional

from pal.storage_manager.components.base import StorageOp, StorageSyscall
from pal.storage_manager.pipeline import StorageManagerPipeline


class StorageSDK:
    def __init__(self, agent_id: str, storage_manager: StorageManagerPipeline):
        self.agent_id = agent_id
        self.storage_manager = storage_manager

    def _submit_and_wait(self, syscall: StorageSyscall):
        request_id = self.storage_manager.submit(syscall)
        try:
            req = self.storage_manager.get_result(request_id)
            while req.status.value == "queued":
                self.storage_manager.tick()
                req = self.storage_manager.get_result(request_id)
            if req.status.value == "error":
                error = req.response.error if req.response else "unknown error"
                raise RuntimeError(f"storage syscall failed: {error}")
            return req.response
        finally:
            self.storage_manager.forget(request_id)

    def read(self, key: str) -> Optional[Any]:
        resp = self._submit_and_wait(StorageSyscall(agent_id=self.agent_id, op=StorageOp.READ, key=key))
        return resp.data

    def write(self, key: str, data: Any) -> None:
        self._submit_and_wait(StorageSyscall(agent_id=self.agent_id, op=StorageOp.WRITE, key=key, data=data))

    def delete(self, key: str) -> None:
        self._submit_and_wait(StorageSyscall(agent_id=self.agent_id, op=StorageOp.DELETE, key=key))

    def list_keys(self, prefix: str = "") -> list[str]:
        resp = self._submit_and_wait(StorageSyscall(agent_id=self.agent_id, op=StorageOp.LIST, key=prefix))
        return resp.data

    def vector_upsert(self, doc_id: str, text: str, metadata: dict = None) -> None:
        self._submit_and_wait(StorageSyscall(
            agent_id=self.agent_id, op=StorageOp.VECTOR_UPSERT, key=doc_id, data=text, metadata=metadata or {},
        ))

    def vector_query(self, query_text: str, top_k: int = 3) -> list[dict]:
        resp = self._submit_and_wait(StorageSyscall(
            agent_id=self.agent_id, op=StorageOp.VECTOR_QUERY, query_text=query_text, top_k=top_k,
        ))
        return resp.data

    # -- cross-agent access, subject to the Access Manager (AIOS 3.8) ---------

    def read_from(self, target_agent_id: str, key: str) -> Optional[Any]:
        resp = self._submit_and_wait(StorageSyscall(
            agent_id=target_agent_id, requesting_agent_id=self.agent_id, op=StorageOp.READ, key=key,
        ))
        return resp.data

    def write_to(self, target_agent_id: str, key: str, data: Any) -> None:
        self._submit_and_wait(StorageSyscall(
            agent_id=target_agent_id, requesting_agent_id=self.agent_id, op=StorageOp.WRITE, key=key, data=data,
        ))

    def delete_from(self, target_agent_id: str, key: str) -> None:
        self._submit_and_wait(StorageSyscall(
            agent_id=target_agent_id, requesting_agent_id=self.agent_id, op=StorageOp.DELETE, key=key,
        ))
