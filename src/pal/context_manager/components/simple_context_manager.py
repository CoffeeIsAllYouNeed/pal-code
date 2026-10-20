"""SimpleContextManager (AIOS Appendix A.4): pause a long generation when its time slice ends and resume it later without
losing the work done.

Two ways, chosen by what the core can give us. Text-based (every cloud API and local server): the text decoded so far is the
snapshot, and resuming continues from it. Logits-based (an in-process Hugging Face model): the beam-search state is the
snapshot. Both are done by the interrupt controller already used by the Round Robin scheduler; this class gives it the
per-caller save, load and clear interface the adapter needs.
"""
from __future__ import annotations

import threading
from typing import Any

from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper


class SimpleContextManager:
    def __init__(self) -> None:
        self._contexts: dict[Any, Any] = {}
        self._lock = threading.Lock()
        from pal.orchestrator.components.context_interrupt import (
            ContextInterruptController,  # lazy: avoids a cycle
        )
        self._controller = ContextInterruptController()

    def load_context(self, pid: Any) -> Any:
        with self._lock:
            return self._contexts.get(pid)

    def clear_context(self, pid: Any) -> None:
        with self._lock:
            self._contexts.pop(pid, None)

    def generate_response_with_interruption(
        self, model_name: str, model: LLMWrapper, messages: list[dict], tools: list[dict] | None, message_return_type: str,
        temperature: float | None, max_tokens: int | None, pid: Any, time_limit: float, response_format: dict | None = None,
    ) -> tuple[str, bool]:
        """Generate for at most `time_limit` seconds. Returns (text so far, finished). If not finished, the state is kept under
        `pid`; calling again with the same pid continues from it. Plain text only: tool calls and JSON output are not interrupted."""
        from pal.orchestrator.components.context_interrupt import ContextSnapshot, InterruptResult

        base = LLMSyscall(messages=messages, model=model_name or None, temperature=temperature, max_tokens=max_tokens)
        snap = self.load_context(pid)
        prior = ""
        if snap is not None and snap.mode == "text":
            prior = snap.text_so_far
            partial = InterruptResult(complete=False, text_so_far=prior, elapsed=0.0, snapshot=snap)
            base = self._controller.build_continuation(base, partial)
        res = self._controller.run_slice(model, base, time_limit, resume_snapshot=snap if snap and snap.mode == "logits" else None)
        text = res.text_so_far if res.cumulative else prior + res.text_so_far
        if res.complete:
            self.clear_context(pid)
            return text, True
        with self._lock:
            self._contexts[pid] = ContextSnapshot(mode=res.snapshot.mode, text_so_far=text, beam_state=res.snapshot.beam_state)
        return text, False
