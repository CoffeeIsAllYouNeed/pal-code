"""
Queued tool-call dispatch with conflict skip-ahead (AIOS section 3.7):
"Request processing involves hashmap verification against usage and
parallel limits; upon detecting conflicts, the system advances to
subsequent queue requests until identifying a conflict-free candidate."

Unlike the LLM syscall scheduler's FIFO/Round-Robin (section 3.3, which
always processes requests in strict arrival or time-sliced order), a
tool-call queue that hits a conflict does NOT block everything behind
it: dispatch_next() scans forward past conflicting requests to find the
first one that's runnable right now and executes that, leaving the
conflicting ones in place (not requeued at the back -- their position
in line doesn't change, only which one runs next does) for a later
dispatch_next() call once capacity frees up.
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pal.tool_manager.components.base import ToolCallRequest, ToolCallResult
from pal.tool_manager.components.conflict_resolver import ToolConflictResolver
from pal.tool_manager.components.param_validator import validate_params
from pal.tool_manager.components.tool_registry import ToolRegistry


@dataclass
class QueuedToolCall:
    request: ToolCallRequest
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: float = field(default_factory=time.time)
    status: str = "queued"  # queued | done | error
    result: Optional[ToolCallResult] = None


class ToolCallQueue:
    def __init__(self, registry: ToolRegistry, conflict_resolver: ToolConflictResolver):
        self.registry = registry
        self.conflicts = conflict_resolver
        self.queue: deque[QueuedToolCall] = deque()
        self.requests: dict[str, QueuedToolCall] = {}

    def submit(self, agent_id: str, tool_name: str, args: dict) -> str:
        request = ToolCallRequest(agent_id=agent_id, tool_name=tool_name, args=args)
        entry = QueuedToolCall(request=request)
        self.requests[entry.id] = entry
        self.queue.append(entry)
        return entry.id

    def dispatch_next(self) -> Optional[QueuedToolCall]:
        """Scans the queue from the front for the first request whose
        tool currently has capacity (hashmap check against its
        parallel_limit), executes it, and removes it from the queue.
        Returns None if the queue is empty or every queued request
        currently conflicts."""
        for entry in list(self.queue):
            spec = self.registry.spec(entry.request.tool_name)
            if not self.conflicts.can_acquire(entry.request.tool_name, spec.parallel_limit):
                continue  # conflict -- advance to the next queued request

            self.queue.remove(entry)
            self._execute(entry)
            return entry

        return None  # nothing runnable right now

    def _execute(self, entry: QueuedToolCall) -> None:
        req = entry.request
        spec = self.registry.spec(req.tool_name)
        self.conflicts.acquire(req.tool_name)
        try:
            validated_args = validate_params(req.tool_name, spec.params, req.args)
            fn = self.registry.get_callable(req.tool_name)
            output = fn(**validated_args)
            entry.result = ToolCallResult(ok=True, result=output)
            entry.status = "done"
        except Exception as e:
            entry.result = ToolCallResult(ok=False, error=str(e))
            entry.status = "error"
        finally:
            self.conflicts.release(req.tool_name)

    def drain(self, max_iterations: int = 10_000) -> list[QueuedToolCall]:
        """Repeatedly dispatch_next() until the queue is empty or
        nothing left is runnable. Since this project executes tools
        synchronously, a slot acquired in _execute() is released again
        before dispatch_next() returns -- so a conflict only actually
        blocks drain() here if the SAME tool's limit was already
        exhausted by an in-flight call held open from OUTSIDE this
        queue (e.g. another agent process using the same
        ToolConflictResolver). max_iterations guards against that
        making no progress."""
        finished = []
        i = 0
        while self.queue and i < max_iterations:
            entry = self.dispatch_next()
            if entry is None:
                break  # every remaining request currently conflicts
            finished.append(entry)
            i += 1
        return finished

    def get_result(self, request_id: str) -> QueuedToolCall:
        return self.requests[request_id]

    def forget(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def pending_count(self) -> int:
        return len(self.queue)
