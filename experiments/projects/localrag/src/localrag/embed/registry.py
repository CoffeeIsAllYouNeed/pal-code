"""Builds the embedder named in the settings. New embedders (a neural model, for example) are added with register_embedder."""
from __future__ import annotations

from collections.abc import Callable

from localrag.core.config import EmbedConfig
from localrag.core.errors import ConfigError
from localrag.embed.base import Embedder
from localrag.embed.cache import CachedEmbedder
from localrag.embed.hashing import HashingEmbedder

_FACTORIES: dict[str, Callable[[EmbedConfig], Embedder]] = {"hashing": lambda cfg: HashingEmbedder(cfg)}


def register_embedder(name: str, factory: Callable[[EmbedConfig], Embedder]) -> None:
    """Make an embedder available under a name that settings can refer to."""
    _FACTORIES[name] = factory


def available_embedders() -> list[str]:
    """The names of the registered embedders, sorted."""
    return sorted(_FACTORIES)


def build_embedder(config: EmbedConfig) -> Embedder:
    """The configured embedder, wrapped in a cache."""
    if config.embedding_name not in _FACTORIES:
        raise ConfigError(f"unknown embedder {config.embedding_name!r}; available: {', '.join(available_embedders())}")
    return CachedEmbedder(_FACTORIES[config.embedding_name](config), config.embedding_cache_size)
