"""The LiteLLM wrapper, the one wrapper for every model call: the cloud APIs (OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together,
OpenRouter, Perplexity, Azure OpenAI, Amazon Bedrock) AND the local runtimes (Ollama, vLLM, LM Studio, llama.cpp servers) all go
through LiteLLM's Python SDK (`litellm.completion`). The only core that does not is the in-process Hugging Face one, because
LiteLLM calls model servers and cannot load a model into this process.

How it is kept safe, because this library handles users' API keys:
  * only the SDK is used, never the LiteLLM proxy server (most of its 2026 CVEs, and the March 2026 backdoor's second payload,
    are in the proxy);
  * the version is pinned and hash-locked (requirements/litellm.lock, pyproject extra `litellm`);
  * the key is passed to each call explicitly, never put in a global;
  * LiteLLM's telemetry and callbacks are off, and its price table is the copy bundled with the pinned version (no download at
    import time).
A local runtime has no key to protect, so for it the point of LiteLLM is one interface, one token and error format, and one
path to test. Stopping a stream early (the scheduler's pause) closes the connection underneath LiteLLM's wrapper, which has no
close() of its own: otherwise the model server would keep generating and block the job that paused it.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from pal.errors import DependencyMissingError
from pal.tokens import estimate_tokens

from .base import DeploymentType, LLMResponse, LLMSyscall, LLMWrapper, WrapperCapabilities, WrapperConfig
from .features import is_response_format, normalize_tools, parse_arguments

PINNED = "1.102.1"

# our provider name -> LiteLLM's provider prefix
PREFIX = {"openai": "openai", "azure_openai": "azure", "anthropic": "anthropic", "gemini": "gemini", "groq": "groq",
          "together": "together_ai", "mistral": "mistral", "deepseek": "deepseek", "openrouter": "openrouter",
          "perplexity": "perplexity", "bedrock": "bedrock"}

# local runtime -> LiteLLM's provider prefix. "ollama_chat" is LiteLLM's /api/chat route (the one that handles chat turns and tools).
LOCAL_PREFIX = {"ollama": "ollama_chat", "vllm": "hosted_vllm", "lmstudio": "lm_studio", "llamacpp": "openai"}
# The OpenAI-style local servers need SOME key in the request. Without an explicit one LiteLLM would fall back to a real key in the
# environment (OPENAI_API_KEY) and send it to a local process, so a placeholder is always sent unless you configured a real one.
_PLACEHOLDER_KEY = "not-needed"
_NEEDS_PLACEHOLDER = ("vllm", "lmstudio", "llamacpp")

_PASSTHROUGH = ("top_p", "stop", "seed", "frequency_penalty", "presence_penalty")
_STREAM_HANDLES = ("streaming_response", "response_iterator", "http_response")
_module: Any = None


def _import_litellm() -> Any:
    """Import LiteLLM once, with its network and logging side effects switched off first."""
    global _module
    if _module is not None:
        return _module
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")      # use the bundled price table: no download at import
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    try:
        import litellm
    except ImportError as e:
        raise DependencyMissingError(
            f"litellm is not installed. Run: pip install 'pal-code[litellm]'  (pinned to litellm=={PINNED}; a hash-locked "
            "install is in requirements/litellm.lock)") from e
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    for attr in ("telemetry",):
        if hasattr(litellm, attr):
            setattr(litellm, attr, False)
    litellm.success_callback = []
    litellm.failure_callback = []
    _module = litellm
    return litellm


def full_model(backend: str, model: str) -> str:
    prefix = PREFIX.get(backend) or LOCAL_PREFIX.get(backend)
    if prefix is None or model.startswith(prefix + "/"):
        return model
    return f"{prefix}/{model}"


def close_stream(stream: Any) -> None:
    """Stop a LiteLLM stream so the model server stops generating. LiteLLM's wrapper has no sync close(); the HTTP stream is held
    by generators underneath it, and closing those closes the connection. Tolerant: anything missing or failing is skipped."""
    inner = getattr(stream, "completion_stream", None)
    for obj in (stream, inner):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001  (already closed, or not closable)
                pass
    for name in _STREAM_HANDLES:
        handle = getattr(inner, name, None)
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


class LiteLLMWrapper(LLMWrapper):
    """Your own key, any provider LiteLLM knows. Two specialised kinds build on it: `OpenRouterWrapper` (one key, hundreds of models) and
    `LocalLLM` (models served on this machine); they set what differs and share everything else, so the key handling lives in one place."""
    name = "litellm"
    IS_LOCAL = False

    def __init__(self, config: WrapperConfig):
        super().__init__(config)
        self.ll = _import_litellm()
        self.model = full_model(config.backend, config.model)
        self.api_key = config.api_key
        self.local = self.IS_LOCAL or config.backend in LOCAL_PREFIX
        self.timeout = float(config.extra.get("timeout", 300 if self.local else 120))   # a local model's first call loads it

    # -- one request ---------------------------------------------------------------------------------------------
    def _kwargs(self, syscall: LLMSyscall) -> dict:
        kw: dict = {"model": syscall.model if syscall.model and "/" in syscall.model else self.model,
                    "messages": syscall.messages, "timeout": self.timeout, "num_retries": 0, "drop_params": True}
        key = self._key()
        if key:
            kw["api_key"] = key
        base = self.config.base_url or self.config.host
        if base:
            kw["api_base"] = base
        if syscall.temperature is not None:
            kw["temperature"] = syscall.temperature
        if syscall.max_tokens is not None:
            kw["max_tokens"] = syscall.max_tokens
        tools = normalize_tools(getattr(syscall, "tools", None))
        if tools:
            kw["tools"] = [{"type": "function", "function": t} for t in tools]
        rf = getattr(syscall, "response_format", None)
        if rf:
            kw["response_format"] = rf if is_response_format(rf) else {
                "type": "json_schema", "json_schema": {"name": "response", "schema": rf}}
        kw.update({k: v for k, v in syscall.params.items() if k in _PASSTHROUGH})
        kw.update(self.config.extra.get("litellm_params", {}))
        return kw

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        t0 = time.time()
        r = self.ll.completion(**self._kwargs(syscall))
        latency = time.time() - t0
        msg = r.choices[0].message
        calls = [{"name": c.function.name, "parameters": parse_arguments(c.function.arguments)}
                 for c in (getattr(msg, "tool_calls", None) or [])]
        usage = getattr(r, "usage", None)
        try:
            cost = float(self.ll.completion_cost(completion_response=r))
        except Exception:  # noqa: BLE001  (a model with no known price)
            cost = None
        return LLMResponse(
            text=getattr(msg, "content", None) or "", raw=r,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            meta={"latency": latency, "tool_calls": calls or None, "cost_usd": cost, "model": self.model})

    def stream_chat(self, syscall: LLMSyscall) -> Iterator[str]:
        stream = self.ll.completion(**self._kwargs(syscall), stream=True)
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    yield piece
        finally:
            close_stream(stream)            # runs when the consumer stops early too (the scheduler pausing a job)

    # -- facts about the model -------------------------------------------------------------------------------------
    def token_count(self, text: str) -> int:
        try:
            return int(self.ll.token_counter(model=self.model, text=text))
        except Exception:  # noqa: BLE001  (an unknown model has no tokenizer: estimate)
            return estimate_tokens(text)

    def _key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return _PLACEHOLDER_KEY if self.config.backend in _NEEDS_PLACEHOLDER else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        kw: dict = {"model": self.model, "input": texts, "timeout": self.timeout}
        key = self._key()
        if key:
            kw["api_key"] = key
        base = self.config.base_url or self.config.host
        if base:
            kw["api_base"] = base
        return [list(d["embedding"]) for d in self.ll.embedding(**kw).data]

    def price_per_token(self) -> tuple[float, float] | None:
        """(input, output) USD per token from LiteLLM's bundled price table, or None for an unknown model. A local model is free."""
        if self.local:
            return None
        try:
            info = self.ll.get_model_info(model=self.model)       # provider-aware: "openrouter/gpt-4o" is not OpenAI's gpt-4o
        except Exception:  # noqa: BLE001  (not in the price table)
            return None
        if not info or info.get("input_cost_per_token") is None:
            return None
        return float(info["input_cost_per_token"]), float(info.get("output_cost_per_token") or 0.0)

    def capabilities(self) -> WrapperCapabilities:
        tools = json_mode = False
        try:
            tools = bool(self.ll.supports_function_calling(model=self.model))
        except Exception:  # noqa: BLE001
            pass
        try:
            json_mode = "response_format" in (self.ll.get_supported_openai_params(model=self.model) or [])
        except Exception:  # noqa: BLE001
            pass
        return WrapperCapabilities(
            deployment_type=DeploymentType.LOCAL_SERVER if self.local else DeploymentType.CLOUD_API, supports_embeddings=self.local,
            supports_streaming=True, supports_tools=tools, supports_json=json_mode, billed_per_token=not self.local)
