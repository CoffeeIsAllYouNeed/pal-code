"""ScheduledWrapper: puts the local model behind the local scheduler without changing the code that uses it.

Anything that calls `core.chat(...)` (the router, the input savers, the verifier) keeps working; the call becomes a job in the
right class. The class comes from the turn that is running (`task(...)`), set once by the chat service.
"""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pal.llm_scheduler.local import LocalScheduler
from pal.llm_scheduler.types import TaskClass, TurnStats
from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper, WrapperCapabilities
from pal.tokens import estimate_tokens

_task: contextvars.ContextVar[tuple[TaskClass, bool]] = contextvars.ContextVar("pal_task", default=(TaskClass.CHAT, False))
_stats: contextvars.ContextVar[TurnStats | None] = contextvars.ContextVar("pal_turn_stats", default=None)


@contextmanager
def task(task_class: TaskClass, preemptible: bool = False) -> Iterator[None]:
    """Everything the local model is asked inside this block belongs to `task_class`. With `preemptible`, a plain-text
    generation may be paused and resumed (only for cores that really stream)."""
    token = _task.set((task_class, preemptible))
    try:
        yield
    finally:
        _task.reset(token)


@contextmanager
def track() -> Iterator[TurnStats]:
    stats = TurnStats()
    token = _stats.set(stats)
    try:
        yield stats
    finally:
        _stats.reset(token)


def current_stats() -> TurnStats | None:
    return _stats.get()


def _streams(core: LLMWrapper) -> bool:
    return type(core).stream_chat is not LLMWrapper.stream_chat


class ScheduledWrapper(LLMWrapper):
    name = "scheduled"

    def __init__(self, core: LLMWrapper, scheduler: LocalScheduler | None):
        self.core = core
        self.scheduler = scheduler
        self.config: Any = getattr(core, "config", None)

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        if self.scheduler is None:
            return self.core.chat(syscall)
        cls, preemptible = _task.get()
        if preemptible and _streams(self.core):
            ticket = self.scheduler.submit_stream(self.core, syscall, cls)
            text = ticket.result()
            prompt = estimate_tokens(" ".join(str(m.get("content", "")) for m in syscall.messages))
            resp = LLMResponse(text=text, prompt_tokens=prompt, completion_tokens=estimate_tokens(text),
                               meta={"streamed": True, "pauses": ticket.pauses})
        else:
            ticket = self.scheduler.submit(lambda: self.core.chat(syscall), cls)
            resp = ticket.result()
        stats = _stats.get()
        if stats is not None:
            stats.add_local(ticket)
        return resp

    def stream_chat(self, syscall: LLMSyscall):
        return self.core.stream_chat(syscall)

    def token_count(self, text: str) -> int:
        return self.core.token_count(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.core.embed(texts)

    def capabilities(self) -> WrapperCapabilities:
        return self.core.capabilities()

    def __getattr__(self, name: str) -> Any:        # e.g. init_beam_search on in-process cores
        if name in ("core", "scheduler", "config"):
            raise AttributeError(name)
        return getattr(self.core, name)
