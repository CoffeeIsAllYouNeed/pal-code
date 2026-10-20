"""
Base abstractions for the LLM core layer (AIOS section 3.2 equivalent).

Every LLM deployment option -- a cloud API, a local server, or a fully
in-process local model -- is wrapped as one "core", the same way a CPU
core abstracts away a physical execution unit from the scheduler that
uses it. Callers never talk to a specific backend's SDK or HTTP schema
directly; they issue one of a small set of *unified system calls*
(LLMSyscall) against whichever core is currently selected, and every
core answers with the same LLMResponse shape.

This is what makes a core "a core" rather than just an adapter class:
the wrapper is not "however OllamaAdapter happens to shape its
send()", it is a fixed contract (chat / token_count / embed /
capabilities) that every deployment option must satisfy, cloud or
local, API-based or in-process.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DeploymentType(str, Enum):
    """How the wrapped LLM instance is actually reached. Mirrors the
    "cloud vs local device, which hardware, which inference framework"
    distinction AIOS 3.2 calls out as the reason a core abstraction is
    needed in the first place."""

    CLOUD_API = "cloud_api"          # billed HTTP API (OpenAI, Anthropic, Gemini, ...)
    LOCAL_SERVER = "local_server"    # local HTTP server (Ollama, llama.cpp, vLLM, LM Studio)
    LOCAL_IN_PROCESS = "local_in_process"  # no server at all -- model loaded into this process


@dataclass
class WrapperConfig:
    """Deployment-agnostic configuration for building any core.

    Not every field is used by every core -- e.g. `device` only matters
    to LOCAL_IN_PROCESS cores, `api_key` is usually unused for local
    ones. Unrecognised/backend-specific settings go in `extra` instead
    of forcing new top-level fields for every new backend.
    """

    backend: str = "ollama"
    model: str = "llama3"
    host: Optional[str] = None          # local server base URL, e.g. http://localhost:11434
    base_url: Optional[str] = None      # cloud/OpenAI-compatible API base URL override
    api_key: Optional[str] = None       # falls back to the backend's standard env var if unset
    device: Optional[str] = None        # "cuda", "cpu", "mps" -- local in-process cores only
    extra: dict = field(default_factory=dict)


@dataclass
class LLMSyscall:
    """A single unified inference request. This is the "system call"
    every core accepts, regardless of deployment option -- callers never
    build a backend-specific payload themselves."""

    messages: list[dict]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    params: dict = field(default_factory=dict)  # backend-specific passthrough, used sparingly


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class WrapperCapabilities:
    """What a given core can actually do, so callers (and the pipeline)
    can branch on capability instead of on backend name."""

    deployment_type: DeploymentType
    supports_kv_cache: bool = False
    supports_embeddings: bool = False
    supports_streaming: bool = False
    supports_tools: bool = False      # native function calling (else the adapter emulates it with a prompt)
    supports_json: bool = False       # native structured (JSON) output (else the adapter emulates it and validates)
    billed_per_token: bool = False   # true for cloud APIs: tokens saved == dollars saved
    # True only for cores with direct access to per-step token logits/
    # search-tree state (in-process local models). Cloud/local-server
    # cores never get raw logits back over HTTP, so this is always
    # False for them -- see orchestrator/components/context_interrupt.py
    # for what this gates: text-based vs logits-based snapshotting.
    supports_logits_snapshot: bool = False


class LLMWrapper(ABC):
    """Abstract wrapper for one LLM instance under one deployment
    option. Concrete cores implement this for a specific backend
    (OpenAI-compatible APIs, Anthropic, Gemini, Ollama, an in-process
    local model, ...) but every one of them is driven through the same
    three system calls below.

    Cores that report capabilities().supports_logits_snapshot=True are
    expected to additionally implement three duck-typed methods used by
    the logits-based context interrupt mechanism (see
    orchestrator/components/context_interrupt.py):
    `init_beam_search(syscall, beam_width)`, `beam_search_step(state,
    max_new_tokens)`, and `decode_beam_search(state)`. These aren't
    declared as abstract methods here because they require direct
    logits access and a concrete search-tree state representation --
    something only an in-process local core (HuggingFaceLocalLLM) can
    provide; every cloud/local-server core simply never sets that
    capability flag and is never asked for them.
    """

    name: str = "base"

    def __init__(self, config: WrapperConfig):
        self.config = config

    # -- unified system calls -------------------------------------------------

    @abstractmethod
    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        """SYS_LLM_CHAT: run one inference request."""
        ...

    def stream_chat(self, syscall: LLMSyscall):
        """SYS_LLM_CHAT_STREAM: same request, but yields text chunks as
        they're generated instead of blocking for the full response.

        This is what the scheduler's Round Robin strategy times against
        for its context-interrupt mechanism (orchestrator/components/
        context_interrupt.py) -- RR needs to be able to stop a generation
        partway through a time slice, not just measure how long a
        blocking call took.

        Default fallback: cores that don't override this yield the
        entire response as one chunk once chat() returns. That's
        functionally correct (RR still works, just can't interrupt
        mid-generation for that core -- the interrupt check only gets
        one opportunity, at the end) but doesn't give RR the granularity
        real token-level streaming does. Cores that expose a real
        streaming endpoint (Ollama, the OpenAI-compatible schema)
        override this with true incremental yielding.
        """
        yield self.chat(syscall).text

    @abstractmethod
    def token_count(self, text: str) -> int:
        """SYS_LLM_TOKEN_COUNT: count tokens the way this core's model
        actually tokenizes, falling back to a heuristic estimate when
        the backend has no tokenizer endpoint available."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """SYS_LLM_EMBED: optional. Cores that don't support embeddings
        raise NotImplementedError rather than silently returning
        zero-vectors, so callers can detect the gap via capabilities()
        instead of getting a wrong answer."""
        raise NotImplementedError(f"{self.name} core does not support embeddings")

    # -- capability / KV-cache introspection ----------------------------------

    @abstractmethod
    def capabilities(self) -> WrapperCapabilities:
        ...

    def supports_kv_cache(self) -> bool:
        return self.capabilities().supports_kv_cache

    def reset_kv_cache(self) -> None:
        pass


class UnavailableWrapper(LLMWrapper):
    """Stands in for a core that could not be set up (for example LiteLLM is not installed). Every call raises, so the callers' own
    "the local model failed" paths run: routing falls back to the API model and the input savers send your message untouched."""

    name = "unavailable"

    def __init__(self, why: str):
        self.why = why

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        raise ConnectionError(f"the local model is unavailable: {self.why}")

    def token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)
