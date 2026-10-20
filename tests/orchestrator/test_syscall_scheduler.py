import time

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMSyscall,
    LLMWrapper,
    WrapperCapabilities,
)
from pal.orchestrator.components.syscall_scheduler import SchedulingStrategy, SyscallScheduler


class FakeStreamingCore(LLMWrapper):
    """A core whose stream_chat() yields a fixed number of chunks, each
    taking `chunk_seconds` to arrive -- long enough to force RR to
    interrupt mid-stream at a small quantum, short enough to keep the
    test itself fast."""

    name = "fake_streaming"

    def __init__(self, config=None, num_chunks: int = 6, chunk_seconds: float = 0.05):
        self.config = config
        self.num_chunks = num_chunks
        self.chunk_seconds = chunk_seconds

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        text = "".join(f"[{i}]" for i in range(self.num_chunks))
        return LLMResponse(text=text)

    def stream_chat(self, syscall: LLMSyscall):
        # A continuation syscall has more messages than the original
        # (the interrupt controller appends 2 per continuation) -- use
        # that to shrink remaining work on resume, so the fake actually
        # finishes eventually instead of looping the full chunk count
        # forever across every resumed slice.
        already_continued = (len(syscall.messages) - 2) // 2
        remaining = max(1, self.num_chunks - already_continued * 2)
        for i in range(remaining):
            time.sleep(self.chunk_seconds)
            yield f"[{i}]"

    def token_count(self, text: str) -> int:
        return len(text.split())

    def capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER, supports_streaming=True)


def test_fifo_runs_each_request_to_completion_in_order():
    scheduler = SyscallScheduler(strategy=SchedulingStrategy.FIFO)
    core = FakeStreamingCore(num_chunks=3, chunk_seconds=0.01)

    id_a = scheduler.submit("agent-a", core, LLMSyscall(messages=[{"role": "user", "content": "a"}]))
    id_b = scheduler.submit("agent-b", core, LLMSyscall(messages=[{"role": "user", "content": "b"}]))

    first = scheduler.tick()
    assert first.id == id_a
    assert first.status.value == "done"

    second = scheduler.tick()
    assert second.id == id_b
    assert second.status.value == "done"


def test_round_robin_interrupts_and_gives_other_requests_a_turn():
    # Quantum shorter than one full generation (6 chunks * 0.05s = 0.3s)
    # forces at least one interruption before either request finishes.
    scheduler = SyscallScheduler(strategy=SchedulingStrategy.ROUND_ROBIN, rr_quantum=0.08)
    core = FakeStreamingCore(num_chunks=6, chunk_seconds=0.05)

    id_a = scheduler.submit("agent-a", core, LLMSyscall(messages=[{"role": "user", "content": "a"}]))
    id_b = scheduler.submit("agent-b", core, LLMSyscall(messages=[{"role": "user", "content": "b"}]))

    # First tick: request A gets a slice, doesn't finish (interrupted),
    # goes back on the queue -- request B, not A, should run next.
    first = scheduler.tick()
    assert first.id == id_a
    assert first.status.value == "queued"  # requeued, not done yet

    second = scheduler.tick()
    assert second.id == id_b  # B got its turn before A was revisited


def test_round_robin_eventually_completes_via_drain():
    scheduler = SyscallScheduler(strategy=SchedulingStrategy.ROUND_ROBIN, rr_quantum=0.08)
    core = FakeStreamingCore(num_chunks=4, chunk_seconds=0.05)

    scheduler.submit("agent-a", core, LLMSyscall(messages=[{"role": "user", "content": "a"}]))
    scheduler.submit("agent-b", core, LLMSyscall(messages=[{"role": "user", "content": "b"}]))

    finished = scheduler.drain(max_ticks=100)
    assert len(finished) == 2
    assert all(r.status.value == "done" for r in finished)
    assert all(r.result is not None for r in finished)
