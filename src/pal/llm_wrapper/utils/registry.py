"""
Backend-name -> wrapper-class registry, and the factory that builds a
concrete LLMWrapper from a WrapperConfig. This is the single place that
knows how a `backend` string maps to a deployment option -- adding a
new cloud API or local runtime means adding one line here, not
touching the pipeline or callers.
"""
from __future__ import annotations

from pal.llm_wrapper.components.base import LLMWrapper, WrapperConfig
from pal.llm_wrapper.components.litellm_wrapper import LOCAL_PREFIX, LiteLLMWrapper
from pal.llm_wrapper.components.litellm_wrapper import PREFIX as LITELLM_PROVIDERS
from pal.llm_wrapper.components.openrouter_wrapper import OpenRouterWrapper
from pal.local_llm.components.huggingface_local import HuggingFaceLocalLLM
from pal.local_llm.components.local_llm import LocalLLM

# EVERY model call goes through ONE wrapper: LiteLLM's Python SDK (see litellm_wrapper.py for how it is kept safe). That covers the
# cloud APIs and the local runtimes. The single exception is the in-process Hugging Face core: LiteLLM calls model servers, it
# cannot load a model into this process.
# EVERY model call goes through ONE wrapper, LiteLLM's Python SDK (see litellm_wrapper.py for how it is kept safe). Three kinds build on it:
#   LiteLLMWrapper    your own key for a provider (OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together, Perplexity, Azure, Bedrock),
#                     or any LiteLLM model string with the backend "litellm"
#   OpenRouterWrapper one OpenRouter key, hundreds of models, a model browser: for exploring
#   LocalLLM          models served on this machine (Ollama, vLLM, LM Studio, llama.cpp), in the Local LLM component
# The single exception is the in-process Hugging Face model, also in the Local LLM component: LiteLLM calls model servers, it cannot load a
# model into this process.
WRAPPER_REGISTRY: dict[str, type[LLMWrapper]] = {
    **{name: LiteLLMWrapper for name in LITELLM_PROVIDERS},
    **{name: LocalLLM for name in LOCAL_PREFIX},
    "openrouter": OpenRouterWrapper,
    "litellm": LiteLLMWrapper,          # any LiteLLM model string, e.g. "cohere/command-r"
    "huggingface_local": HuggingFaceLocalLLM,
}
# Names the paper's backend table uses, mapped to ours.
BACKEND_ALIASES = {"google": "gemini", "huggingface": "huggingface_local", "hf": "huggingface_local",
                   "aws_bedrock": "bedrock"}


def build_wrapper(config: WrapperConfig) -> LLMWrapper:
    config.backend = BACKEND_ALIASES.get(config.backend, config.backend)
    cls = WRAPPER_REGISTRY.get(config.backend)
    if cls is None:
        known = ", ".join(sorted(WRAPPER_REGISTRY))
        raise ValueError(f"unknown backend: {config.backend!r} (known backends: {known})")
    return cls(config)


def available_backends() -> list[str]:
    return sorted(WRAPPER_REGISTRY)
