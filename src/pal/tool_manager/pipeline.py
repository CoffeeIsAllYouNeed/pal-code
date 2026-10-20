"""
Tool Manager (AIOS section 3.7): standardized tool loading (uniform
interface, parameter validation before execution, dynamic
dependency-checked loading by name) and resolution of tool-call
conflicts for tools with parallel-access limits (a hashmap of
real-time in-flight counts, and a queue that skips ahead past
conflicting requests rather than blocking on them).
"""
from typing import Optional

from pal.tool_manager.components.base import ToolCallResult, ToolParam, ToolSpec
from pal.tool_manager.components.conflict_resolver import ToolConflictResolver
from pal.tool_manager.components.tool_call_queue import QueuedToolCall, ToolCallQueue
from pal.tool_manager.components.tool_registry import ToolRegistry


class ToolManagerPipeline:
    def __init__(self):
        self.registry = ToolRegistry()
        self.conflicts = ToolConflictResolver()
        self.queue = ToolCallQueue(self.registry, self.conflicts)

    def register_tool(
        self,
        name: str,
        loader,
        params: list[ToolParam] = None,
        parallel_limit: Optional[int] = None,
        dependencies: list[str] = None,
    ) -> None:
        self.registry.register(ToolSpec(
            name=name,
            loader=loader,
            params=params or [],
            parallel_limit=parallel_limit,
            dependencies=dependencies or [],
        ))

    def list_tools(self) -> list[str]:
        return self.registry.list_tools()

    def invoke(self, agent_id: str, tool_name: str, args: dict) -> ToolCallResult:
        """Synchronous convenience call: submits to the queue and drains
        dispatch attempts until THIS specific request completes (i.e.
        until it's the conflict-free candidate dispatch_next() picks).
        Looks synchronous to the caller even though it went through the
        same centralized queue every tool call does -- same shape as
        StorageSDK and Pipeline's LLM syscall dispatch."""
        if not self.registry.is_registered(tool_name):
            return ToolCallResult(ok=False, error=f"unknown tool: {tool_name}")
        request_id = self.queue.submit(agent_id, tool_name, args)
        try:
            entry = self.queue.get_result(request_id)
            while entry.status == "queued":
                dispatched = self.queue.dispatch_next()
                if dispatched is None:
                    # Every queued call currently conflicts -- nothing this
                    # loop can do would make progress right now.
                    return ToolCallResult(
                        ok=False,
                        error=f"tool '{tool_name}' could not be dispatched -- all queued calls currently conflict",
                    )
                entry = self.queue.get_result(request_id)
            return entry.result
        finally:
            self.queue.forget(request_id)

    # -- queue-driven path, for callers that want direct control ---------------

    def submit(self, agent_id: str, tool_name: str, args: dict) -> str:
        return self.queue.submit(agent_id, tool_name, args)

    def dispatch_next(self) -> Optional[QueuedToolCall]:
        return self.queue.dispatch_next()

    def drain(self, max_iterations: int = 10_000) -> list[QueuedToolCall]:
        return self.queue.drain(max_iterations=max_iterations)

    def get_result(self, request_id: str) -> QueuedToolCall:
        return self.queue.get_result(request_id)
