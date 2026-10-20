"""Types for the LLM adapter (AIOS Appendix A.2): the config of one LLM instance, the unified inference syscall, the
unified response, and the routing strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, WrapperConfig

ACTIONS = ("chat", "chat_with_json_output", "chat_with_tool_call_output")


class RouterStrategy(str, Enum):
    """How the adapter picks one of several LLM instances for a request."""

    Sequential = "sequential"      # take the instances in turn, request after request


@dataclass
class LLMConfig:
    """One LLM instance: which model, on which backend, reached how."""

    name: str                       # the model name
    backend: str                    # openai, anthropic, gemini, groq, bedrock, huggingface_local, vllm, ollama, ...
    max_new_tokens: int = 1000
    temperature: float | None = None   # None = leave to the call (or the adapter's default)
    hostname: str | None = None     # server URL for local backends, or an API base URL override
    api_key: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        d = dict(d)
        return cls(
            name=d.pop("name", None) or d.pop("model", None) or "",
            backend=d.pop("backend", ""),
            max_new_tokens=int(d.pop("max_new_tokens", d.pop("max_tokens", 1000))),
            temperature=(float(d.pop("temperature")) if d.get("temperature") is not None else d.pop("temperature", None)),
            hostname=d.pop("hostname", None) or d.pop("host", None) or d.pop("base_url", None),
            api_key=d.pop("api_key", None),
            extra=d)

    def to_core_config(self) -> WrapperConfig:
        return WrapperConfig(backend=self.backend, model=self.name, host=self.hostname, base_url=self.hostname,
                          api_key=self.api_key, device=self.extra.get("device"), extra=dict(self.extra))


@dataclass
class LLMAdapterSyscall(LLMSyscall):
    """The unified LLM inference system call. Fields beyond a plain chat request are real attributes (not `params`),
    because cores pass `params` straight into the provider's HTTP payload."""

    action_type: str = "chat"                    # chat | chat_with_json_output | chat_with_tool_call_output
    tools: list[dict] | None = None
    message_return_type: str = "text"            # text | json
    response_format: dict | None = None          # JSON schema (or {"type": "json_object"}) for structured output
    llms: list[str] | None = None                # restrict this call to these named LLMs
    pid: Any = None                              # identifies the caller for interrupt and resume
    time_limit: float | None = None              # seconds; with a context manager, generation can be interrupted


@dataclass
class LLMAdapterResponse:
    """The unified response. `response_message` is text or a JSON string; `tool_calls` is a list of
    {"name": ..., "parameters": {...}}; a failure is reported here, never raised."""

    response_message: str | None = None
    tool_calls: list[dict] | None = None
    finished: bool = False
    error: str | None = None
    status_code: int = 200
    # not in the paper: Pal needs these to count tokens and time
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    backend: str = ""
    latency: float = 0.0

    @property
    def text(self) -> str:
        return self.response_message or ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_core_response(self) -> LLMResponse:
        return LLMResponse(text=self.text, prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                           meta={"latency": self.latency, "tool_calls": self.tool_calls, "finished": self.finished})
