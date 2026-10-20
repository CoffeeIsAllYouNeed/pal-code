from dataclasses import dataclass

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMSyscall,
    LLMWrapper,
    WrapperCapabilities,
)
from pal.orchestrator.components.context_interrupt import ContextInterruptController
from pal.orchestrator.components.syscall_scheduler import SchedulingStrategy, SyscallScheduler


@dataclass
class FakeBeamState:
    """Stands in for HuggingFaceLocalLLM's real BeamSearchState (which
    needs torch/transformers, not available in this test environment).
    Same shape of information: a step counter as a stand-in for
    "generated tokens so far", and a finished flag."""

    step: int = 0
    finished: bool = False
    total_steps: int = 6


class FakeLogitsCore(LLMWrapper):
    """A core reporting supports_logits_snapshot=True and implementing
    the duck-typed init_beam_search / beam_search_step / decode_beam_search
    trio, so ContextInterruptController's logits-based path can be
    exercised without a real model."""

    name = "fake_logits"

    def __init__(self, config=None, total_steps: int = 6):
        self.config = config
        self.total_steps = total_steps

    def chat(self, syscall: LLMSyscall):
        raise NotImplementedError("this test only exercises the logits-based streaming path")

    def token_count(self, text: str) -> int:
        return len(text.split())

    def capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(
            deployment_type=DeploymentType.LOCAL_IN_PROCESS,
            supports_logits_snapshot=True,
        )

    def init_beam_search(self, syscall: LLMSyscall, beam_width: int = 1) -> FakeBeamState:
        return FakeBeamState(step=0, finished=False, total_steps=self.total_steps)

    def beam_search_step(self, state: FakeBeamState, max_new_tokens: int = 512) -> FakeBeamState:
        new_step = state.step + 1
        return FakeBeamState(step=new_step, finished=new_step >= state.total_steps, total_steps=state.total_steps)

    def decode_beam_search(self, state: FakeBeamState) -> str:
        return f"step-{state.step}"


def test_logits_path_selected_for_capable_core():
    controller = ContextInterruptController()
    core = FakeLogitsCore(total_steps=3)
    syscall = LLMSyscall(messages=[{"role": "user", "content": "hi"}])

    result = controller.run_slice(core, syscall, quantum=999)  # huge quantum -> runs to completion in one slice
    assert result.complete
    assert result.cumulative is True
    assert result.snapshot.mode == "logits"
    assert result.text_so_far == "step-3"


def test_logits_path_interrupts_and_resumes_via_beam_state_not_text():
    controller = ContextInterruptController()
    core = FakeLogitsCore(total_steps=100)  # long enough to force an interrupt at quantum=0
    syscall = LLMSyscall(messages=[{"role": "user", "content": "hi"}])

    first = controller.run_slice(core, syscall, quantum=0)  # interrupt after the very first step
    assert not first.complete
    assert first.snapshot.mode == "logits"
    assert first.snapshot.beam_state.step == 1

    # Resume from the snapshot -- syscall.messages is untouched (unlike
    # text mode, nothing was rebuilt into the prompt).
    second = controller.run_slice(core, syscall, quantum=0, resume_snapshot=first.snapshot)
    assert second.snapshot.beam_state.step == 2
    assert syscall.messages == [{"role": "user", "content": "hi"}]  # unchanged


def test_scheduler_round_robin_drains_logits_based_request():
    scheduler = SyscallScheduler(strategy=SchedulingStrategy.ROUND_ROBIN, rr_quantum=0)  # 1 beam step per tick
    core = FakeLogitsCore(total_steps=4)

    scheduler.submit("agent-a", core, LLMSyscall(messages=[{"role": "user", "content": "hi"}]))

    finished = scheduler.drain(max_ticks=50)
    assert len(finished) == 1
    req = finished[0]
    assert req.status.value == "done"
    assert req.result.text == "step-4"
    assert req.slices_run >= 4  # at least one tick per beam step given quantum=0
