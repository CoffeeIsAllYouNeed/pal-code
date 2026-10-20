"""The model layer. Its names are loaded on first use (PEP 562), so importing a light submodule such as pal.llm_wrapper.modes (the provider
names and key variables) does not load the whole model stack: pal cli draws its first screen before any of it is needed."""
from importlib import import_module

_WHERE = {
    "LLMAdapter": "pal.llm_wrapper.adapter",
    "LLMAdapterResponse": "pal.llm_wrapper.components.adapter_types",
    "LLMAdapterSyscall": "pal.llm_wrapper.components.adapter_types",
    "LLMConfig": "pal.llm_wrapper.components.adapter_types",
    "RouterStrategy": "pal.llm_wrapper.components.adapter_types",
    "DeploymentType": "pal.llm_wrapper.components.base",
    "LLMResponse": "pal.llm_wrapper.components.base",
    "LLMSyscall": "pal.llm_wrapper.components.base",
    "LLMWrapper": "pal.llm_wrapper.components.base",
    "WrapperCapabilities": "pal.llm_wrapper.components.base",
    "WrapperConfig": "pal.llm_wrapper.components.base",
    "LLMWrapperPipeline": "pal.llm_wrapper.pipeline",
    "available_backends": "pal.llm_wrapper.utils.registry",
    "build_wrapper": "pal.llm_wrapper.utils.registry",
}

__all__ = list(_WHERE)


def __getattr__(name: str):
    if name in _WHERE:
        value = getattr(import_module(_WHERE[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pal.llm_wrapper' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_WHERE))
