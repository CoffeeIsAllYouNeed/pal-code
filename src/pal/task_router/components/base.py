from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Prefix on assistant turns that the local LLM produced. It lives in the shared
# session history, so every later API request shows which turns were the local
# model's work rather than the API model's own.
LOCAL_TAG = "[local-llm]"

# Added to the API system prompt (only once the session holds a local turn).
API_NOTICE = (
    f"Turns tagged {LOCAL_TAG} were done by a small local model before you, not by you. "
    "Their results are the current state of the work: build on them, and correct them only if wrong."
)


@dataclass
class RouteDecision:
    route: str            # "local" | "api"
    reason: str
    source: str           # "override" | "gate" | "classifier" | "fallback" | "disabled" | "escalated"


@dataclass
class LocalAnswer:
    text: Optional[str]   # None = the local model declined; go to the API model
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def has_local_turns(history: list[dict]) -> bool:
    return any(m.get("role") == "assistant" and str(m.get("content", "")).startswith(LOCAL_TAG) for m in history)
