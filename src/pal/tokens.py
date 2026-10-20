"""Token counting. The default is a local estimate: it costs nothing and never touches the
network. `provider_counter` asks the model's own tokenizer endpoint, which is exact but adds a
network round trip per call, so it is opt-in (settings.token_counter = "provider")."""
from __future__ import annotations

from collections.abc import Callable


def estimate_tokens(text: str) -> int:
    """~4 characters per token, at least 1 for non-empty text."""
    return max(1, len(text) // 4) if text else 0


def estimate_messages(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in messages)


def make_counter(mode: str, core=None) -> Callable[[str], int]:
    if mode == "provider" and core is not None:
        return core.token_count
    return estimate_tokens
