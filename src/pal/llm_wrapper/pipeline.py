"""
LLMWrapperPipeline is the single entry point callers use to talk to an LLM,
regardless of deployment option. It owns exactly one active LLMWrapper at a
time (swappable via switch()) and exposes the unified system calls --
callers never see LiteLLMWrapper vs the in-process Hugging Face core vs a local
in-process model, only syscall_chat/syscall_token_count/syscall_embed.

This replaces the old Router, which only dispatched to a fixed set of
BackendAdapter subclasses with backend-specific interfaces; the syscall
layer here is the "standardized system calls" piece AIOS 3.2 describes
and the old Router didn't actually have.
"""
from __future__ import annotations

from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, WrapperCapabilities, WrapperConfig
from pal.llm_wrapper.utils.registry import available_backends, build_wrapper


class LLMWrapperPipeline:
    def __init__(self, config: WrapperConfig):
        self.config = config
        self.core = build_wrapper(config)

    def switch(self, config: WrapperConfig) -> None:
        """Swap the active core -- e.g. moving from a local Ollama core
        to a cloud Gemini core mid-session. Every syscall below is core-
        agnostic, so nothing else needs to change when this is called."""
        self.config = config
        self.core = build_wrapper(config)

    # -- unified system calls -------------------------------------------------

    def syscall_chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        syscall = LLMSyscall(
            messages=messages,
            model=kwargs.pop("model", None),
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            stream=kwargs.pop("stream", False),
            params=kwargs,
        )
        return self.core.chat(syscall)

    def syscall_token_count(self, text: str) -> int:
        return self.core.token_count(text)

    def syscall_embed(self, texts: list[str]) -> list[list[float]]:
        return self.core.embed(texts)

    def capabilities(self) -> WrapperCapabilities:
        return self.core.capabilities()

    def supports_kv_cache(self) -> bool:
        return self.core.supports_kv_cache()

    def reset_kv_cache(self) -> None:
        self.core.reset_kv_cache()

    @staticmethod
    def list_backends() -> list[str]:
        return available_backends()
