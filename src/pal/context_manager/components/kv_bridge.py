from pal.llm_wrapper.components.base import LLMWrapper


class KVBridge:
    """Delegates dedup to the active LLM core's native KV cache when
    available (e.g. llama.cpp's prompt-prefix cache, Ollama's automatic
    prefix reuse), instead of duplicating that logic here."""

    def __init__(self, core: LLMWrapper):
        self.core = core

    def active(self) -> bool:
        return self.core.supports_kv_cache()

    def reset(self) -> None:
        self.core.reset_kv_cache()
