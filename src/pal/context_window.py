"""How many tokens a model can take in one request (its context window), for the "tokens left" counters in the terminal and the web page.

API models: the size LiteLLM knows for the model, else a safe default. Local models: the context Pal gives Ollama (num_ctx), which is what
really limits them here, whatever the model could do on paper."""
from __future__ import annotations

from functools import lru_cache

LOCAL_CONTEXT = 8192            # Pal runs local models with num_ctx 8192 (pal.terminal.backend.make_core)
API_DEFAULT = 128_000           # a model LiteLLM does not know: most current API models take at least this
LOCAL_PROVIDERS = ("local", "ollama", "lmstudio", "vllm", "llamacpp", "llama.cpp", "huggingface")


@lru_cache(maxsize=256)
def context_window(provider: str, model: str) -> int:
    provider = (provider or "").lower()
    if provider in LOCAL_PROVIDERS:
        return LOCAL_CONTEXT
    if not model:
        return API_DEFAULT
    try:
        import litellm
        for name in (f"{provider}/{model}" if provider else model, model, model.split("/")[-1]):
            try:
                info = litellm.get_model_info(name)
            except Exception:  # noqa: BLE001  (not in LiteLLM's table under this name)
                continue
            size = info.get("max_input_tokens") or info.get("max_tokens")
            if size:
                return int(size)
    except Exception:  # noqa: BLE001  (LiteLLM missing or broken: the default)
        pass
    return API_DEFAULT


def tokens_left(window: int, last_prompt: int) -> int:
    """What is left of the window after the last request's input (the next one is usually at least as big)."""
    return max(0, window - max(0, last_prompt))


def short(n: int) -> str:
    return f"{n / 1000:.1f}k".replace(".0k", "k") if n >= 10_000 else f"{n:,}"
