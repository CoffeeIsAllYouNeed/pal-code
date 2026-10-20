import tempfile

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMWrapper,
    WrapperCapabilities,
    WrapperConfig,
)
from pal.orchestrator.components.syscall_scheduler import SchedulingStrategy, SyscallScheduler
from pal.pipeline import Pipeline


class FakeCore(LLMWrapper):
    name = "fake"

    def __init__(self, config=None, tag="x"):
        self.config = config
        self.tag = tag

    def chat(self, syscall):
        return LLMResponse(text=f"reply-from-{self.tag}", prompt_tokens=3, completion_tokens=2)

    def token_count(self, text):
        return len(text.split())

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


def test_two_pipelines_share_one_centralized_scheduler():
    """Two separate Pipeline instances (standing in for two separate
    agents/modules) submit through the SAME SyscallScheduler -- proving
    request queueing is centralized rather than each Pipeline running
    its own private queue."""
    shared_scheduler = SyscallScheduler(strategy=SchedulingStrategy.FIFO)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline_a = Pipeline(
            WrapperConfig(backend="ollama", model="llama3"),
            memory_store_path=f"{tmp}/mem_a.json",
            syscall_scheduler=shared_scheduler,
            agent_name="agent-a",
        )
        pipeline_a.llm_wrapper.core = FakeCore(tag="a")

        pipeline_b = Pipeline(
            WrapperConfig(backend="ollama", model="llama3"),
            memory_store_path=f"{tmp}/mem_b.json",
            syscall_scheduler=shared_scheduler,
            agent_name="agent-b",
        )
        pipeline_b.llm_wrapper.core = FakeCore(tag="b")

        out_a = pipeline_a.run("system prompt", "hello from a")
        out_b = pipeline_b.run("system prompt", "hello from b")

        assert out_a == "reply-from-a"
        assert out_b == "reply-from-b"

        # Both requests actually went through the one shared scheduler,
        # not a private queue each Pipeline owned itself.
        assert len(shared_scheduler.requests) == 2
        agent_names = {req.agent_name for req in shared_scheduler.requests.values()}
        assert agent_names == {"agent-a", "agent-b"}
