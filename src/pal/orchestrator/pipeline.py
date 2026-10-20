from typing import TYPE_CHECKING, Optional

from pal.access_manager.components.base import AccessDecision, OperationType
from pal.access_manager.pipeline import AccessManagerPipeline
from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper
from pal.orchestrator.components.agent import AgentConfig
from pal.orchestrator.components.cgroup_manager import CgroupManager, create_resource_manager
from pal.orchestrator.components.scheduler import Scheduler
from pal.orchestrator.components.syscall_scheduler import SchedulingStrategy, SyscallRequest, SyscallScheduler
from pal.orchestrator.utils.device import ToolDevice
from pal.storage_manager.components.base import StorageSyscall
from pal.storage_manager.components.storage_queue import StorageRequest
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.storage_manager.utils.sdk import StorageSDK
from pal.tool_manager.components.base import ToolCallResult, ToolParam
from pal.tool_manager.components.tool_call_queue import QueuedToolCall
from pal.tool_manager.pipeline import ToolManagerPipeline

if TYPE_CHECKING:
    from pal.memory.pipeline import MemoryPipeline


class OrchestratorPipeline:
    """
    Top-level entry point for agent orchestration: submit agent configs,
    admit them onto a bounded number of concurrent cgroup-isolated
    processes, and optionally expose a tool device those agent
    processes can call back into.

    Also owns the centralized syscall queue (SyscallScheduler): every
    LLM inference request in the system, from any module, is meant to
    go through submit_syscall() here rather than each caller hitting an
    LLMWrapper directly -- that's what "centralize all queues within the
    scheduler module" means in practice (AIOS 3.3).

    And owns the Storage Manager (AIOS 3.6): one StorageManagerPipeline
    shared by every agent/pipeline in the system, for both the memory
    manager's RAM-to-disk swaps (see pal/memory/pipeline.py's
    storage_manager= parameter) and an agent's own direct storage SDK
    calls (storage_sdk_for()).

    And owns the Tool Manager (AIOS 3.7): one ToolManagerPipeline
    (standardized loading, parameter validation, parallel-access
    conflict resolution), wired into the ToolDevice transport so every
    tool call an agent makes over the socket gets the same treatment
    regardless of which agent or which tool.

    And owns the Access Manager (AIOS 3.8): one AccessManagerPipeline,
    wired into the StorageManagerPipeline so every cross-agent storage
    request (StorageSDK.read_from()/write_to()/delete_from()) is
    checked against a privilege-group hashmap, and every destructive
    operation (delete, overwrite) goes through user intervention.
    Agents are assigned a privilege group via assign_privilege_group()
    or submit_agent(..., privilege_group=...).

    And provides the AIOS SDK (AIOS 3.9) via sdk_for(agent_id,
    core_config): one AgentSDK per agent, wired to every manager above,
    which is the actual interface an agent developer builds against --
    chat()/remember()/recall()/read()/write()/call_tool()/etc. -- rather
    than any manager or syscall type directly.
    """

    def __init__(
        self,
        parent_slice: str = "pal.slice",
        max_concurrent: int = 3,
        tool_socket_path: str = None,  # a Unix socket path, or host:port (used on Windows)
        syscall_strategy: SchedulingStrategy = SchedulingStrategy.FIFO,
        rr_quantum: float = 2.0,
        storage_root: str = None,
        vector_persist_directory: str = None,
        confirmation_handler=None,
        resource_manager=None,
    ):
        self.cgroup_manager = resource_manager or create_resource_manager(parent_slice, manager_cls=CgroupManager)
        self.scheduler = Scheduler(self.cgroup_manager, max_concurrent=max_concurrent)
        self.syscall_scheduler = SyscallScheduler(strategy=syscall_strategy, rr_quantum=rr_quantum)
        self.access_manager = AccessManagerPipeline(confirmation_handler=confirmation_handler)
        self.storage_manager = StorageManagerPipeline(
            root=storage_root, vector_persist_directory=vector_persist_directory, access_manager=self.access_manager
        )
        self.tool_manager = ToolManagerPipeline()
        self.tool_device = ToolDevice(tool_socket_path, tool_manager=self.tool_manager) if tool_socket_path \
            else ToolDevice(tool_manager=self.tool_manager)

    def submit_agent(self, config: AgentConfig, priority: int = 0, weight: int = 100, privilege_group: str = None) -> None:
        self.scheduler.submit(config, priority=priority, weight=weight)
        if privilege_group is not None:
            self.access_manager.assign_privilege_group(config.name, privilege_group)

    def tick(self) -> list[str]:
        return self.scheduler.tick()

    def run_turn(self, agent_name: str, system_prompt: str, user_message: str, timeout: float = 60) -> dict:
        agent = self.scheduler.running.get(agent_name)
        if agent is None:
            raise ValueError(f"agent {agent_name} is not running -- submit + tick first")
        return agent.run_turn(system_prompt, user_message, timeout=timeout)

    # -- centralized syscall scheduling (AIOS 3.3) -----------------------------

    def submit_syscall(self, agent_name: str, core: LLMWrapper, syscall: LLMSyscall, priority: int = 0) -> str:
        """Every LLM inference request funnels through here rather than
        callers hitting core.chat()/core.stream_chat() directly, so
        request ordering and time-slicing are handled in exactly one
        place regardless of which module -- an Agent, the top-level
        Pipeline, anything else -- issued the request."""
        return self.syscall_scheduler.submit(agent_name, core, syscall, priority=priority)

    def syscall_tick(self) -> SyscallRequest | None:
        """Advances the syscall queue by one step, using whichever
        strategy this orchestrator was configured with (FIFO or Round
        Robin)."""
        return self.syscall_scheduler.tick()

    def drain_syscalls(self, max_ticks: int = 10_000) -> list[SyscallRequest]:
        return self.syscall_scheduler.drain(max_ticks=max_ticks)

    def syscall_result(self, request_id: str) -> SyscallRequest:
        return self.syscall_scheduler.get_result(request_id)

    # -- Storage Manager (AIOS 3.6) --------------------------------------------

    def storage_sdk_for(self, agent_id: str) -> StorageSDK:
        """What an agent calls to read/write its own persistent data --
        "the agent calls the storage API in the SDK, which is further
        converted into storage-related system calls and put into the
        storage queue." Every StorageSDK instance shares this
        orchestrator's one StorageManagerPipeline/queue, scoped by
        agent_id."""
        return StorageSDK(agent_id, self.storage_manager)

    def submit_storage_syscall(self, syscall: StorageSyscall) -> str:
        return self.storage_manager.submit(syscall)

    def storage_tick(self) -> StorageRequest | None:
        return self.storage_manager.tick()

    def drain_storage_syscalls(self, max_iterations: int = 10_000) -> list[StorageRequest]:
        return self.storage_manager.drain(max_iterations=max_iterations)

    def storage_result(self, request_id: str) -> StorageRequest:
        return self.storage_manager.get_result(request_id)

    # -- Access Manager (AIOS 3.8) ----------------------------------------------

    def assign_privilege_group(self, agent_id: str, group: str) -> None:
        self.access_manager.assign_privilege_group(agent_id, group)

    def set_confirmation_handler(self, handler) -> None:
        """Registers the callback that answers user-intervention
        prompts for irreversible operations (delete, overwrite,
        privilege modification). The default handler denies everything,
        so a real application needs to set one -- e.g. a CLI y/n prompt
        or a UI confirmation dialog."""
        self.access_manager.set_confirmation_handler(handler)

    def check_access(self, requesting_agent_id: str, target_agent_id: str, operation: OperationType, resource: str = "") -> AccessDecision:
        return self.access_manager.check_access(requesting_agent_id, target_agent_id, operation, resource=resource)

    def register_tool(self, name: str, loader, params: list[ToolParam] = None,
                       parallel_limit: int = None, dependencies: list[str] = None) -> None:
        """Registers a tool with the Tool Manager (AIOS 3.7): `loader`
        is called at most once, the first time this tool is actually
        invoked (dynamic loading -- see ToolRegistry.get_callable).
        Reachable both via the ToolDevice socket transport and directly
        through invoke_tool() below."""
        self.tool_device.register_tool(
            name, loader, params=params, parallel_limit=parallel_limit, dependencies=dependencies
        )

    def invoke_tool(self, agent_id: str, tool_name: str, args: dict) -> ToolCallResult:
        return self.tool_manager.invoke(agent_id, tool_name, args)

    def submit_tool_call(self, agent_id: str, tool_name: str, args: dict) -> str:
        return self.tool_manager.submit(agent_id, tool_name, args)

    def dispatch_tool_call(self) -> QueuedToolCall | None:
        return self.tool_manager.dispatch_next()

    def drain_tool_calls(self, max_iterations: int = 10_000) -> list[QueuedToolCall]:
        return self.tool_manager.drain(max_iterations=max_iterations)

    # -- AIOS SDK (AIOS 3.9) -----------------------------------------------------

    def sdk_for(self, agent_id: str, core_config, memory: Optional["MemoryPipeline"] = None):
        """Builds an AgentSDK for one agent, wired to this
        orchestrator's shared managers (storage, tools, access control)
        and centralized LLM syscall scheduler -- the single object an
        agent developer actually interacts with, abstracting away every
        manager and syscall type above it. `core_config` is
        agent-specific (different agents may run against different
        backends/models), everything else is shared."""
        from pal.llm_wrapper.pipeline import LLMWrapperPipeline
        from pal.sdk.pipeline import AgentSDK

        llm_wrapper = LLMWrapperPipeline(core_config)
        return AgentSDK(
            agent_id=agent_id,
            llm_wrapper=llm_wrapper,
            storage_manager=self.storage_manager,
            tool_manager=self.tool_manager,
            access_manager=self.access_manager,
            memory=memory,
            syscall_scheduler=self.syscall_scheduler,
        )

    def register_builtin_tools(self) -> None:
        """Opt-in: registers the SDK's small built-in tool catalog
        (pal/sdk/components/tool_catalog.py) with this
        orchestrator's ToolManagerPipeline. Not called automatically --
        a fresh OrchestratorPipeline starts with no tools registered,
        exactly as before this existed; call this explicitly to make
        the built-in catalog available."""
        from pal.sdk.components.tool_catalog import register_builtin_tools
        register_builtin_tools(self.tool_manager)

    def start_tool_device(self) -> None:
        self.tool_device.start()

    def stop_tool_device(self) -> None:
        self.tool_device.stop()

    def status(self) -> dict:
        return self.scheduler.status()

    def stop_all(self) -> None:
        self.scheduler.stop_all()
