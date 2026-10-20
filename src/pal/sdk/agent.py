"""Native agent development: what a developer subclasses to build an agent for Pal.

A *native* agent is written for Pal: it talks to the system only through the SDK's pre-defined APIs
(`self.sdk.chat`, `.remember`, `.recall`, `.write`, `.read`, `.search_knowledge`, `.call_tool`, `.request_access`).
It never holds a model client, a file handle or a tool directly, so Pal can route, trim, cache, limit and
measure every call. A *non-native* agent is written with another framework and is adapted instead
(see `pal.sdk.adapt`).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from pal.errors import UserInputError

_READ_TOOLS = frozenset({"read_source", "search_documents", "read_document"})


@dataclass(frozen=True)
class AgentSpec:
    """The small, declarative part of an agent."""

    id: str
    name: str
    description: str
    role: str                     # kept short: it is sent with every request
    key_env: str = ""             # environment variable holding this agent's API key ("" = none needed)
    tools: frozenset[str] = _READ_TOOLS
    max_tokens: int = 1024        # output cap per call (free tiers are small)


@dataclass
class AgentResult:
    answer: str
    ok: bool = True
    steps: list[Any] = field(default_factory=list)   # ReAct steps, Reflexion trials, ...
    error: str = ""
    meta: dict = field(default_factory=dict)


class Kernel(Protocol):
    """Anything that runs a full token-saving turn: ChatService does."""

    def send(self, session_id: str, provider: str, model: str, api_key: str | None, message: str,
             attachments: list[dict], *args: Any, **kw: Any) -> Any: ...


class NativeAgent:
    """Base class for native agents.

    `sdk` gives the pre-defined APIs. `kernel` (optional) is the token-saving turn pipeline; when present, the
    default `run()` uses it, so the agent gets input trimming, document and code-graph retrieval, paged history
    and output rules for free. Without a kernel, `run()` makes one plain SDK call with the agent's role.
    Override `run()` to write your own logic with the SDK calls (see ReActAgent and ReflexionAgent).
    """

    spec: ClassVar[AgentSpec | None] = None

    def __init__(self, sdk: Any = None, kernel: Kernel | None = None):
        self.sdk = sdk
        self.kernel = kernel

    def run(self, task: str, attachments: list[dict] | None = None, session: str = "agent") -> AgentResult:
        spec = self.spec
        if spec is None:
            raise NotImplementedError("a native agent needs a `spec` or its own run()")
        if self.kernel is not None:
            res = self.kernel.send(session, "", "", None, task, list(attachments or []), agent=spec.id)
            return AgentResult(res.reply, meta={"input_tokens": res.input_tokens, "output_tokens": res.output_tokens})
        if self.sdk is None:
            raise RuntimeError("no SDK or kernel: create the agent with sdk= or kernel=")
        text = self.sdk.chat([{"role": "system", "content": spec.role}, {"role": "user", "content": task}],
                             max_tokens=spec.max_tokens)
        return AgentResult(text)


def agent_key(spec: AgentSpec, env: Mapping[str, str] | None = None) -> str | None:
    """The agent's own key, else the shared GEMINI_API_KEY."""
    env = os.environ if env is None else env
    return (env.get(spec.key_env) if spec.key_env else None) or env.get("GEMINI_API_KEY") or None


def get_spec(agents: Mapping[str, type[NativeAgent]], agent_id: str) -> AgentSpec:
    cls = agents.get(agent_id)
    if cls is None or cls.spec is None:
        raise UserInputError(f"unknown agent {agent_id!r}; choose one of: {', '.join(agents)}")
    return cls.spec


def list_agents(agents: Mapping[str, type[NativeAgent]], env: Mapping[str, str] | None = None) -> list[dict]:
    out = []
    for cls in agents.values():
        s = cls.spec
        if s is not None:
            out.append({"id": s.id, "name": s.name, "description": s.description,
                        "ready": bool(agent_key(s, env)), "key_env": s.key_env})
    return out
