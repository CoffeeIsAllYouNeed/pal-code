"""
Agent framework adapters (AIOS section 3.9): let agents built with an
existing framework (Autogen, Open Interpreter, MetaGPT, ...) run inside
pal without any modification to the agent's own code. Each
adapter locates that framework's core LLM-completion function and
monkeypatches it to redirect through an AgentSDK's chat() instead -- so
the framework keeps its own agent-creation API, but every actual model
call goes through pal's llm_wrapper (compression, caching, paging,
the syscall scheduler, all of it) rather than the framework's own HTTP
client.

These adapters target each framework's documented core completion
entry point. A framework's internals can and do change between
releases, so install() raises a clear FrameworkAdapterError (naming the
missing attribute) rather than failing silently if a targeted attribute
path isn't found -- an adapter that silently no-ops would leave an
agent quietly calling the real API instead of pal, which is worse
than a loud failure that tells the caller exactly what broke.
"""
from abc import ABC, abstractmethod

from pal.sdk.pipeline import AgentSDK


class FrameworkAdapterError(RuntimeError):
    pass


class FrameworkAdapter(ABC):
    framework_name: str = "base"

    def __init__(self, sdk: AgentSDK):
        self.sdk = sdk
        self._installed = False

    @abstractmethod
    def install(self) -> None:
        """Monkeypatches the framework's core completion function to
        redirect through self.sdk.chat(). Raises FrameworkAdapterError
        if the framework isn't installed, or if its expected entry
        point isn't found (version mismatch)."""
        ...

    @abstractmethod
    def uninstall(self) -> None:
        """Restores the framework's original completion function."""
        ...

    @property
    def installed(self) -> bool:
        return self._installed

    @staticmethod
    def _check_free(fn) -> None:
        """Refuse to patch a function another adapter already replaced: two SDKs cannot share one framework."""
        if getattr(fn, "_pal_adapter", False):
            raise FrameworkAdapterError("this framework is already adapted by another SDK; uninstall that one first")

    @staticmethod
    def _mark(fn):
        fn._pal_adapter = True
        return fn

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()
