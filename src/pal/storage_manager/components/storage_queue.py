"""
Centralized storage system-call queue (AIOS section 3.6): "the agent
calls the storage API in the SDK, which is further converted into
storage-related system calls and put into the storage queue by the
scheduler. The storage manager then processes the requests in the
queue."

Mirrors how llm_wrapper' inference requests are centralized in
orchestrator/components/syscall_scheduler.py (section 3.3) -- one
queue, submit() to add work, something drives it forward. Storage ops
are effectively instantaneous local file/DB operations (no
multi-second generation to time-slice), so this queue only needs FIFO
processing; there's no equivalent need for Round Robin / context-
interrupt machinery here.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pal.storage_manager.components.base import StorageResponse, StorageSyscall


class StorageSyscallStatus(str, Enum):
    QUEUED = "queued"
    DONE = "done"
    ERROR = "error"


@dataclass
class StorageRequest:
    syscall: StorageSyscall
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: float = field(default_factory=time.time)
    status: StorageSyscallStatus = StorageSyscallStatus.QUEUED
    response: Optional[StorageResponse] = None


class StorageSyscallQueue:
    def __init__(self, pipeline):
        # `pipeline` is a StorageManagerPipeline, kept untyped here to
        # avoid a circular import between this module and pipeline.py
        # (which owns the queue and passes itself in).
        self.pipeline = pipeline
        self.queue: deque[StorageRequest] = deque()
        self.requests: dict[str, StorageRequest] = {}
        self._lock = threading.Lock()

    def submit(self, syscall: StorageSyscall) -> str:
        req = StorageRequest(syscall=syscall)
        with self._lock:
            self.requests[req.id] = req
            self.queue.append(req)
        return req.id

    def process_next(self) -> Optional[StorageRequest]:
        with self._lock:
            if not self.queue:
                return None
            req = self.queue.popleft()
        try:
            req.response = self.pipeline.execute(req.syscall)
            req.status = StorageSyscallStatus.DONE if req.response.ok else StorageSyscallStatus.ERROR
        except Exception as e:
            req.response = StorageResponse(ok=False, error=str(e))
            req.status = StorageSyscallStatus.ERROR
        return req

    def drain(self, max_iterations: int = 10_000) -> list[StorageRequest]:
        finished = []
        i = 0
        while self.queue and i < max_iterations:
            finished.append(self.process_next())
            i += 1
        return finished

    def get_result(self, request_id: str) -> StorageRequest:
        return self.requests[request_id]

    def forget(self, request_id: str) -> None:
        """Drop a finished request. Callers that wait synchronously (the SDK) call this so the
        request table does not grow for the life of the process."""
        self.requests.pop(request_id, None)

    def pending_count(self) -> int:
        return len(self.queue)
