"""Shared fallback token-count heuristic, used when a core's real
tokenizer/endpoint is unavailable (offline, no matching tiktoken
encoding, network error on a pure estimation call, etc.)."""


def heuristic_token_count(text: str) -> int:
    """~4 characters per token is the same rough approximation every
    core already falls back to individually; centralised here so it's
    defined once."""
    return max(1, len(text) // 4)
