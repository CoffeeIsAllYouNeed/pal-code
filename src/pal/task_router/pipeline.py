"""Task routing: easy tasks are done by the local LLM, the rest go to the API model.

Order of decisions for each chat message:
  1. Override ("local" / "api" from the UI) wins.
  2. Deterministic gates send obvious hard input (attachments, code, long text) to the API.
  3. The local LLM classifies what is left; unclear or failed = API.
  4. A task routed local is answered by the local LLM, which may still ESCALATE to the API.

API agents are told about local work through the shared session history: local
answers are stored tagged with LOCAL_TAG, and the API system prompt carries a
one-line notice once such a turn exists (see ChatService).
"""
from __future__ import annotations

from typing import Optional

from pal.config import TaskRoutingConfig
from pal.llm_wrapper.components.base import LLMWrapper
from pal.task_router.components import answerer, classifier, gates
from pal.task_router.components.base import (
    API_NOTICE,
    LOCAL_TAG,
    LocalAnswer,
    RouteDecision,
    has_local_turns,
)

__all__ = ["TaskRouter", "RouteDecision", "LocalAnswer", "LOCAL_TAG", "API_NOTICE", "has_local_turns"]

MODES = ("auto", "local", "api")


class TaskRouter:
    def __init__(self, core: Optional[LLMWrapper], cfg=None):
        self.core = core
        self.cfg = TaskRoutingConfig.coerce(cfg)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def default_mode(self) -> str:
        return self.cfg.default

    def decide(self, message: str, attachments: list, mode: Optional[str] = None,
               history: Optional[list] = None) -> RouteDecision:
        if not self.enabled:
            return RouteDecision("api", "task routing is off", "disabled")
        mode = mode if mode in MODES else self.default_mode()
        if mode in ("local", "api"):
            return RouteDecision(mode, f"{mode} only was chosen", "override")
        blocked = gates.gate(message, attachments, self.cfg)
        if blocked:
            return blocked
        recent = ""
        if history and self.cfg.history_messages:
            recent = "\n".join(f"{m['role']}: {str(m['content'])[:200]}" for m in history[-2:])
        return classifier.classify(self.core, message, recent)

    def answer(self, message: str, history: list[dict], attachments: Optional[list] = None) -> LocalAnswer:
        return answerer.answer(self.core, message, history, self.cfg.history_messages, attachments)
