"""
Standardized tool interface and call request/response shapes (AIOS
section 3.7). Every tool the tool manager loads conforms to ToolSpec --
callers never invoke a tool's underlying callable directly, only ever
through ToolManagerPipeline.invoke(), which validates parameters and
resolves parallel-access conflicts uniformly regardless of what the
tool itself actually does.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParam:
    name: str
    type: type
    required: bool = True
    default: Any = None


@dataclass
class ToolSpec:
    """A tool's standardized description. `loader` is called at most
    once, the first time the tool is invoked by name -- "dynamically
    loads the tool instance including executables initialization" means
    exactly this: constructing/initializing whatever the tool actually
    needs (a client, a subprocess handle, a loaded model, ...) is
    deferred until the tool is first used, not paid for every tool ever
    registered."""

    name: str
    loader: Callable[[], Callable[..., Any]]  # returns the callable that actually runs the tool
    params: list[ToolParam] = field(default_factory=list)
    parallel_limit: Optional[int] = None       # None = unlimited concurrent calls
    dependencies: list[str] = field(default_factory=list)  # importable module names required at load time


@dataclass
class ToolCallRequest:
    agent_id: str
    tool_name: str
    args: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: float = field(default_factory=time.time)


@dataclass
class ToolCallResult:
    ok: bool
    result: Any = None
    error: Optional[str] = None
