"""One entry point for non-native agents: `adapt("autogen", sdk)`.

Third-party frameworks are adapted by replacing the framework's own core model-call function so the call goes to
the SDK instead; the agent's code is not changed. ReAct and Reflexion are reasoning patterns, not libraries, so
they ship as native agents (`ReActAgent`, `ReflexionAgent`).

    sdk = AgentSDK.standalone(WrapperConfig(backend="gemini", model="...", api_key="..."))
    with adapt("autogen", sdk):
        ...  # your existing AutoGen code, unchanged
"""
from __future__ import annotations

from pal.sdk.components.autogen_adapter import AutoGenAdapter
from pal.sdk.components.framework_adapter_base import FrameworkAdapter, FrameworkAdapterError
from pal.sdk.components.metagpt_adapter import MetaGPTAdapter
from pal.sdk.components.open_interpreter_adapter import OpenInterpreterAdapter
from pal.sdk.pipeline import AgentSDK

_ADAPTERS: dict[str, type[FrameworkAdapter]] = {
    "autogen": AutoGenAdapter, "open-interpreter": OpenInterpreterAdapter, "metagpt": MetaGPTAdapter}
_ALIASES = {"open_interpreter": "open-interpreter", "interpreter": "open-interpreter", "pyautogen": "autogen"}
_PATTERNS = {"react": "ReActAgent", "reflexion": "ReflexionAgent"}


def _norm(name: str) -> str:
    n = name.strip().lower()
    return _ALIASES.get(n, n)


def adapt(framework: str, sdk: AgentSDK) -> FrameworkAdapter:
    """An adapter for `framework`. Use it as a context manager, or call install() and uninstall() yourself."""
    n = _norm(framework)
    if n in _PATTERNS:
        raise FrameworkAdapterError(f"{framework} is a reasoning pattern, not a library: use {_PATTERNS[n]}(sdk) "
                                    "from pal.sdk")
    cls = _ADAPTERS.get(n)
    if cls is None:
        raise FrameworkAdapterError(f"no adapter for {framework!r}; supported: {', '.join(supported_frameworks())}")
    return cls(sdk)


def supported_frameworks() -> list[str]:
    return [*_ADAPTERS, *_PATTERNS]
