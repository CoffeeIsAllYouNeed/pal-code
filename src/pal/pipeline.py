import time
import uuid

from pal.agent_constraints.pipeline import AgentConstraintsPipeline
from pal.compression.pipeline import CompressionPipeline
from pal.context_manager.components.pager import Segment
from pal.context_manager.pipeline import ContextManagerPipeline
from pal.llm_wrapper.components.base import LLMSyscall, WrapperConfig
from pal.llm_wrapper.pipeline import LLMWrapperPipeline
from pal.memory.pipeline import MemoryPipeline
from pal.metrics.pipeline import MetricsPipeline


class Pipeline:
    """
    One full chat turn, top to bottom:
    user message -> compression + memory recall -> cache + paging ->
    an LLM core (any deployment option) -> compressed reply, with
    metrics recorded and history/memory updated.

    This is the class every other entry point (webapp, orchestrator
    agents, experiments) drives -- it's the composition root for every
    pal/<component>/pipeline.py above.
    """

    def __init__(
        self,
        core_config: WrapperConfig,
        session_id: str = None,
        memory_store_path: str = None,
        syscall_scheduler=None,
        agent_name: str = None,
        storage_manager=None,
    ):
        self.llm_wrapper = LLMWrapperPipeline(core_config)
        self.context_manager = ContextManagerPipeline(self.llm_wrapper.core)
        self.compression = CompressionPipeline()
        self.agent_constraints = AgentConstraintsPipeline()
        self.metrics = MetricsPipeline()
        self.history: list[Segment] = []
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_name = agent_name or self.session_id

        # Persistent, cross-session memory (survives restarts) -- distinct
        # from context_manager's pager, which only pages within THIS open
        # conversation and forgets everything once the process exits.
        # When a shared StorageManagerPipeline is supplied (AIOS 3.6 --
        # see pal/storage_manager/), the memory manager's disk
        # swaps route through it, scoped by this Pipeline's agent_name;
        # otherwise MemoryPipeline falls back to its own private,
        # unscoped local store.
        self.storage_manager = storage_manager
        self.memory = MemoryPipeline(
            store_path=memory_store_path,
            agent_id=self.agent_name,
            storage_manager=storage_manager,
        )

        # Optional centralized syscall scheduler (AIOS 3.3 -- see
        # orchestrator/components/syscall_scheduler.py). When one is
        # supplied (typically the same SyscallScheduler an
        # OrchestratorPipeline owns, shared across every agent/pipeline
        # in the system), this Pipeline submits its LLM requests there
        # instead of calling the core directly -- so ordering and
        # time-slicing across MULTIPLE pipelines/agents actually goes
        # through one queue, not a private one per Pipeline instance.
        # Left as None by default (e.g. the standalone webapp use case),
        # in which case run() falls back to calling llm_wrapper directly.
        self.syscall_scheduler = syscall_scheduler

    # kept for callers that reached into the old Router/adapter directly
    # (e.g. the webapp's naive-vs-optimized benchmark path)
    @property
    def core(self):
        return self.llm_wrapper.core

    def run(self, system_prompt: str, user_message: str, **kwargs) -> str:
        # Permanent output-behavior constraints, merged into every system
        # prompt so the model generates tersely up front rather than
        # being trimmed after the fact.
        system_prompt = self.agent_constraints.with_constraints(system_prompt)

        compressed_msg = self.compression.compress_input(user_message)

        # Persistent memory recall: pull relevant notes from PAST
        # sessions (not this conversation's own history). Injected into
        # the system prompt so it doesn't distort the naive-vs-optimized
        # token comparison for history paging below.
        recalled_memories = self.memory.retrieve(compressed_msg, top_k=3)
        if recalled_memories:
            memory_block = "\n".join(f"- {n.content}" for n in recalled_memories)
            system_prompt = f"{system_prompt}\n\nRelevant memory from past sessions:\n{memory_block}"

        cache_hit, _ = self.context_manager.cache_system_prompt(system_prompt)
        paged = self.context_manager.page_history(self.history, compressed_msg)

        messages = [{"role": "system", "content": system_prompt}]
        for seg in paged:
            messages.append({"role": seg.role, "content": seg.content})
        messages.append({"role": "user", "content": compressed_msg})

        # Naive baseline: what a naive/unbounded implementation would
        # send this turn -- the FULL history (not paged/trimmed) plus
        # system prompt and new message. Comparing it against what was
        # actually sent (optimized_tokens below) reflects real paging
        # savings, not a same-payload measured-two-ways artifact.
        naive_messages = [system_prompt] + [seg.content for seg in self.history] + [compressed_msg]
        naive_tokens = sum(self.llm_wrapper.syscall_token_count(t) for t in naive_messages)

        t0 = time.time()
        response = self._dispatch_chat(messages, **kwargs)
        latency = time.time() - t0

        output_text = self.compression.compress_output(response.text)

        # Optimized figure: the server's own reported prompt-eval count,
        # when available. For local server cores this reflects tokens
        # actually reprocessed -- if the backend's KV cache reused a
        # matching prefix, this number comes back lower than the naive
        # estimate on its own. Falls back to the estimate otherwise
        # (meaning: no measurable saving, not a caching failure).
        optimized_tokens = response.prompt_tokens if response.prompt_tokens else naive_tokens

        self.metrics.record(
            tokens_naive=naive_tokens,
            tokens_optimized=optimized_tokens,
            latency=latency,
            cache_hit=cache_hit,
        )

        self.history.append(Segment(id=str(len(self.history)), content=compressed_msg, role="user"))
        self.history.append(Segment(id=str(len(self.history)), content=output_text, role="assistant"))

        # Persist this turn to long-term memory so it can be recalled in
        # a FUTURE session (a future Pipeline instance, possibly after a
        # restart) -- not just later in this same conversation.
        self.memory.add(compressed_msg, session_id=self.session_id, tags=["user_turn"])

        return output_text

    def _dispatch_chat(self, messages: list[dict], **kwargs):
        """Routes the actual inference call either through the shared
        centralized SyscallScheduler (when one was supplied -- request
        ordering/time-slicing then happens there, alongside every other
        pipeline/agent sharing that scheduler) or directly through
        llm_wrapper when running standalone."""
        if self.syscall_scheduler is None:
            return self.llm_wrapper.syscall_chat(messages, **kwargs)

        syscall = LLMSyscall(
            messages=messages,
            model=kwargs.pop("model", None),
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            stream=kwargs.pop("stream", False),
            params=kwargs,
        )
        request_id = self.syscall_scheduler.submit(self.agent_name, self.llm_wrapper.core, syscall)

        # Drive the shared scheduler until THIS request reaches a
        # terminal state. Other pipelines/agents sharing the same
        # scheduler may have requests interleaved into these ticks too
        # (that's the point of centralizing) -- this loop just doesn't
        # return control to this Pipeline until its own turn is done.
        req = self.syscall_scheduler.get_result(request_id)
        while req.status.value in ("queued", "running"):
            self.syscall_scheduler.tick()
            req = self.syscall_scheduler.get_result(request_id)

        if req.status.value == "error":
            raise RuntimeError(f"syscall failed: {req.error}")
        return req.result

    def summary(self) -> dict:
        """Metrics summary including cache memory overhead, so savings
        can be checked against the cost of the caching layer itself."""
        return self.metrics.summary(cache_overhead_bytes=self.context_manager.size_bytes())
