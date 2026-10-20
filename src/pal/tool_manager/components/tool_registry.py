"""
Standardized tool loading (AIOS section 3.7): tools are registered by
ToolSpec but not actually constructed until first invoked by name --
"dynamically loads the tool instance including executables
initialization and dependencies verification" happens exactly once,
lazily, here, and the resulting callable is cached for every later
call.
"""
import importlib

from pal.tool_manager.components.base import ToolSpec


class ToolDependencyError(ImportError):
    pass


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._loaded: dict[str, object] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"no tool registered with name '{name}'")
        return self._specs[name]

    def get_callable(self, name: str):
        """Dynamically loads (once) and returns the tool's underlying
        callable. Verifies every declared dependency imports cleanly
        BEFORE running the tool's own loader, so a missing dependency
        fails with a clear ToolDependencyError naming exactly which
        module is missing, rather than an opaque ImportError raised
        from somewhere inside the tool's own loader/initialization
        code."""
        if name in self._loaded:
            return self._loaded[name]

        spec = self.spec(name)
        for dep in spec.dependencies:
            try:
                importlib.import_module(dep)
            except (ImportError, OSError) as e:
                raise ToolDependencyError(f"tool '{name}' requires '{dep}', which is not installed") from e

        instance = spec.loader()
        self._loaded[name] = instance
        return instance

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def list_tools(self) -> list[str]:
        return sorted(self._specs.keys())
