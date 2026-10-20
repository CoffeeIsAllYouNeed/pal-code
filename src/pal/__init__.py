"""Pal: a model-agnostic chat layer that cuts tokens and latency.

Importing the package is cheap on purpose. The older `Pipeline` (which pulls in the compression
stack: nltk, spacy, ...) is loaded only if you ask for it.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    for name in ("pal-code", "pal-ai", "pal-aios"):      # the current name first, then earlier ones
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.1.5"           # a source checkout that is not installed


__version__ = _version()

__all__ = ["Pipeline", "WrapperConfig", "__version__"]

_LAZY = {"Pipeline": "pal.pipeline", "WrapperConfig": "pal.llm_wrapper.components.base"}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pal' has no attribute {name!r}")
