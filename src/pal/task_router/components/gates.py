"""Deterministic checks that run before the classifier. They can only send a
task to the API model, never to the local one: "easy" needs the classifier's
agreement, "hard" is obvious from the shape of the input."""
from __future__ import annotations

from typing import Optional

from pal.task_router.components.base import RouteDecision


def gate(message: str, attachments: list, cfg) -> Optional[RouteDecision]:
    text = message.strip()
    if not text:
        return RouteDecision("api", "empty message", "gate")
    if attachments:
        return RouteDecision("api", "attached files go to the API model", "gate")
    if "```" in text:
        return RouteDecision("api", "contains a code block", "gate")
    if len(text) > cfg.max_local_chars:
        return RouteDecision("api", "message too long for the local model", "gate")
    if text.count("\n") + 1 > cfg.max_local_lines:
        return RouteDecision("api", "message has too many lines", "gate")
    return None
