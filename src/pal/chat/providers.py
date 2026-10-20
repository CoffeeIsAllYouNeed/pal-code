"""API providers the chat can talk to. Local models are never chat models: they only route, reduce
and verify. API keys are used in memory for the request and never logged or stored."""
from __future__ import annotations

import os

from pal.errors import UserInputError
from pal.llm_wrapper.components.base import WrapperConfig
from pal.llm_wrapper.modes import PROVIDER_KEYS
from pal.llm_wrapper.utils.registry import build_wrapper

API_PROVIDERS = PROVIDER_KEYS          # the table lives with the two modes (llm_wrapper/modes.py)


def build_api_core(provider: str, model: str, api_key: str | None = None):
    if provider not in API_PROVIDERS:
        raise UserInputError(f"unsupported provider {provider!r}; API models only: {', '.join(API_PROVIDERS)}")
    if not model:
        raise UserInputError("choose a model")
    key = api_key or os.environ.get(API_PROVIDERS[provider])
    if not key:
        raise UserInputError(f"no API key: paste one in settings or set {API_PROVIDERS[provider]}")
    return build_wrapper(WrapperConfig(backend=provider, model=model, api_key=key))
