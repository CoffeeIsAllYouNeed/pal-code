"""
AIOS SDK (AIOS section 3.9): what an agent developer actually imports
and uses. Wraps every kernel manager behind one small, agent-facing
interface -- chat(), remember()/recall(), read()/write()/delete(),
call_tool(), search_knowledge(), request_access() -- so building an
agent means calling straightforward methods, not assembling
LLMSyscalls, StorageSyscalls, and ToolCallRequests by hand. Every
method here is a thin pass-through to the corresponding manager; the
abstraction IS not having to know those managers exist, matching "this
SDK... abstracts complex system calls, allowing developers to focus on
the agent's internal workflows."

One AgentSDK instance is scoped to exactly one agent_id -- every
storage/memory/tool call it makes is attributed to that agent, and
every access-manager check runs as that agent. OrchestratorPipeline.
sdk_for() is the usual way to get one, wired to the orchestrator's
shared managers.
"""
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from pal.access_manager.components.base import AccessDecision, OperationType
from pal.access_manager.pipeline import AccessManagerPipeline
from pal.errors import PalError, UserInputError
from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper
from pal.llm_wrapper.pipeline import LLMWrapperPipeline
from pal.memory.pipeline import MemoryPipeline
from pal.os_kernel import OSKernel
from pal.sdk.os_api import OSAPI
from pal.sdk.queries import (
    OS_KERNEL,
    PAL_KERNEL,
    LLMQuery,
    MemoryQuery,
    OSQuery,
    Query,
    Response,
    StorageQuery,
    ToolQuery,
)
from pal.sdk.savers import CallSavers
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.storage_manager.utils.sdk import StorageSDK
from pal.tool_manager.pipeline import ToolManagerPipeline


class AgentSDK:
    def __init__(
        self,
        agent_id: str,
        llm_wrapper: Optional[LLMWrapperPipeline],
        storage_manager: StorageManagerPipeline,
        tool_manager: ToolManagerPipeline,
        access_manager: AccessManagerPipeline,
        memory: Optional[MemoryPipeline] = None,
        syscall_scheduler=None,
        savers: Optional[CallSavers] = None,
        os_kernel: Optional[OSKernel] = None,
    ):
        self.agent_id = agent_id
        self.savers = savers
        self._os_kernel = os_kernel
        self.os = OSAPI(self)
        self._llm_wrapper = llm_wrapper
        self._syscall_scheduler = syscall_scheduler
        self._storage = StorageSDK(agent_id, storage_manager)
        self._tool_manager = tool_manager
        self._access_manager = access_manager
        self.memory = memory or MemoryPipeline(agent_id=agent_id, storage_manager=storage_manager)

    @classmethod
    def standalone(cls, core, agent_id: str = "agent", root: Optional[str] = None,
                   savers: Optional[CallSavers] = None, os_kernel: Optional[OSKernel] = None) -> "AgentSDK":
        """A ready-to-use SDK with no orchestrator: one call to start using Pal from your own agent code.
        `core` is an LLMWrapper or a WrapperConfig. Storage lives under `root` (default ~/.pal/sdk). Built-in tools
        (calculator, time, word count) are registered. Light savers (tidy) are on; the exact-repeat cache is off."""
        from pal.llm_wrapper.components.base import WrapperConfig
        from pal.llm_wrapper.utils.registry import build_wrapper
        from pal.paths import data_dir
        from pal.sdk.components.tool_catalog import register_builtin_tools

        active = build_wrapper(core) if isinstance(core, WrapperConfig) else core
        base = root or data_dir("sdk")
        storage = StorageManagerPipeline(root=str(base))
        tools = ToolManagerPipeline()
        register_builtin_tools(tools)
        access = AccessManagerPipeline()
        kernel = os_kernel or OSKernel(access=access, root=Path(base) / "workspaces")
        return cls(agent_id, SimpleNamespace(core=active), storage, tools, access,  # type: ignore[arg-type]
                   savers=savers if savers is not None else CallSavers(), os_kernel=kernel)

    @property
    def stats(self) -> dict:
        """Calls, cache hits, tidied characters, provider-reported tokens and time spent in model calls."""
        s = self.savers.stats if self.savers else None
        return {k: getattr(s, k) for k in ("calls", "cache_hits", "chars_tidied", "prompt_tokens",
                                           "completion_tokens", "seconds")} if s else {}

    # -- chat / inference -------------------------------------------------------

    def chat_response(self, messages: list[dict], core: Optional[LLMWrapper] = None, **kwargs) -> LLMResponse:
        """Runs one LLM inference call and returns the full response (text plus token counts).
        Under the hood this is an LLM syscall -- through the shared centralized SyscallScheduler when
        one was supplied (AIOS 3.3), or directly against the core otherwise. `core` overrides the
        SDK's own core for this call (e.g. a per-request API model chosen in the UI)."""
        active = core or (self._llm_wrapper.core if self._llm_wrapper is not None else None)
        if active is None:
            raise RuntimeError("no LLM core: pass core= or construct the SDK with llm_wrapper")
        sv = self.savers
        key = None
        if sv is not None:
            messages = sv.prepare(messages)
            key = sv.key(getattr(getattr(active, "config", None), "model", ""), messages, kwargs)
            cached = sv.get(key)
            if cached is not None:
                sv.stats.calls += 1
                return cached
        started = time.perf_counter()
        resp = self._infer(active, messages, kwargs)
        if sv is not None:
            sv.stats.calls += 1
            sv.stats.prompt_tokens += resp.prompt_tokens or 0
            sv.stats.completion_tokens += resp.completion_tokens or 0
            sv.stats.seconds += time.perf_counter() - started
            sv.put(key or "", resp)
        return resp

    def _infer(self, active: LLMWrapper, messages: list[dict], kwargs: dict) -> LLMResponse:
        syscall = LLMSyscall(messages=messages, **kwargs)
        if self._syscall_scheduler is not None:
            request_id = self._syscall_scheduler.submit(self.agent_id, active, syscall)
            req = self._syscall_scheduler.get_result(request_id)
            while req.status.value in ("queued", "running"):
                self._syscall_scheduler.tick()
                req = self._syscall_scheduler.get_result(request_id)
            if req.status.value == "error":
                raise RuntimeError(f"chat failed: {req.error}")
            return req.result
        return active.chat(syscall)

    def chat(self, messages: list[dict], core: Optional[LLMWrapper] = None, **kwargs) -> str:
        """Like chat_response(), returning only the reply text."""
        return self.chat_response(messages, core=core, **kwargs).text

    # -- query routing: LLM queries to the Pal kernel, non-LLM queries to the OS kernel ----------------------
    def send_request(self, query: Query) -> Response:
        """The one place that decides where a query goes, by its class alone:
        LLMQuery to the Pal kernel (model), Memory/Storage/Tool queries to the Pal kernel's managers, OSQuery to the
        OS kernel (Ubuntu: files, network, system facts). A refusal or failure comes back as Response(ok=False)."""
        cls = getattr(query, "query_class", "")
        dest = OS_KERNEL if cls == "os" else PAL_KERNEL
        try:
            if isinstance(query, LLMQuery):
                kw = {k: v for k, v in (("max_tokens", query.max_tokens), ("temperature", query.temperature)) if v is not None}
                return Response(True, self.chat_response(query.messages, **kw), handled_by=dest)
            if isinstance(query, MemoryQuery):
                data = self.remember(**query.params) if query.op == "remember" else self.recall(**query.params)
                return Response(True, data, handled_by=dest)
            if isinstance(query, StorageQuery):
                return Response(True, self._storage_op(query), handled_by=dest)
            if isinstance(query, ToolQuery):
                return Response(True, self.call_tool(query.tool, **query.args), handled_by=dest)
            if isinstance(query, OSQuery):
                if self._os_kernel is None:
                    raise UserInputError("this SDK has no OS kernel interface")
                return Response(True, self._os_kernel.handle(self.agent_id, query.op, query.params), handled_by=dest)
        except (PalError, RuntimeError, ValueError, TypeError, KeyError) as e:
            return Response(False, error=str(e), handled_by=dest)
        raise UserInputError(f"unknown query class {cls!r}")

    def _storage_op(self, q: StorageQuery) -> Any:
        ops = {"read": self.read, "write": self.write, "delete": self.delete, "list": self.list_keys,
               "add_knowledge": self.add_knowledge, "search_knowledge": self.search_knowledge}
        if q.op not in ops:
            raise UserInputError(f"unknown storage operation {q.op!r}")
        return ops[q.op](**q.params)

    # -- memory (AIOS 3.5) --------------------------------------------------------

    def remember(self, content: str, tags: Optional[list[str]] = None, session_id: Optional[str] = None):
        return self.memory.add(content, session_id=session_id or self.agent_id, tags=tags)

    def recall(self, query: str, top_k: int = 3):
        return self.memory.retrieve(query, top_k=top_k)

    # -- storage: own data (AIOS 3.6) ---------------------------------------------

    def read(self, key: str) -> Optional[Any]:
        return self._storage.read(key)

    def write(self, key: str, data: Any) -> None:
        self._storage.write(key, data)

    def delete(self, key: str) -> None:
        self._storage.delete(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        return self._storage.list_keys(prefix=prefix)

    def add_knowledge(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        self._storage.vector_upsert(doc_id, text, metadata=metadata)

    def search_knowledge(self, query: str, top_k: int = 3) -> list[dict]:
        return self._storage.vector_query(query, top_k=top_k)

    # -- storage: another agent's data, subject to the Access Manager (3.8) ------

    def read_from(self, target_agent_id: str, key: str) -> Optional[Any]:
        return self._storage.read_from(target_agent_id, key)

    def write_to(self, target_agent_id: str, key: str, data: Any) -> None:
        self._storage.write_to(target_agent_id, key, data)

    def delete_from(self, target_agent_id: str, key: str) -> None:
        self._storage.delete_from(target_agent_id, key)

    def request_access(self, target_agent_id: str, operation: OperationType, resource: str = "") -> AccessDecision:
        """Checks (without performing) whether this agent would be
        allowed to touch target_agent_id's resource -- useful for an
        agent to decide whether to even attempt a cross-agent call
        before issuing it."""
        return self._access_manager.check_access(self.agent_id, target_agent_id, operation, resource=resource)

    # -- tools (AIOS 3.7) -----------------------------------------------------------

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        result = self._tool_manager.invoke(self.agent_id, tool_name, kwargs)
        if not result.ok:
            raise RuntimeError(f"tool '{tool_name}' failed: {result.error}")
        return result.result

    def list_tools(self) -> list[str]:
        return self._tool_manager.list_tools()

    def close(self) -> None:
        """Delete this agent's stored files and vectors (a chat session ended). Memory notes are
        kept: they are the cross-session part."""
        self._storage.storage_manager.purge_agent(self.agent_id)
