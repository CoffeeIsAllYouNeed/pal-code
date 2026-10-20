"""The OpenRouter wrapper: one OpenRouter key reaches hundreds of models from many labs, some of them free. It is for EXPLORING models: trying a
new one costs nothing but a model name. (If you have your own key for one provider, use the LiteLLM wrapper directly: no middleman.)

It is the LiteLLM wrapper underneath, so the key is handled exactly the same way (per call, never in a global), plus:
  * the app's name is sent as OpenRouter's optional attribution header;
  * `models()` reads OpenRouter's public model list (no key needed) so you can browse: name, context size, price, and whether it is free.

What to know before using it: your prompts pass through OpenRouter on their way to the model's lab; free models are rate-limited (about 20 a
minute); the list changes, and a model can disappear. The list is read live and kept for an hour; if it cannot be read you get an error, never
a stale guess.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pal.llm_wrapper.components.litellm_wrapper import LiteLLMWrapper

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
CATALOGUE_TTL = 3600.0
APP_TITLE = "Pal"


@dataclass(frozen=True)
class ModelInfo:
    id: str                          # what you pass as the model name, e.g. "meta-llama/llama-3.3-70b-instruct:free"
    name: str
    context_length: int | None
    prompt_per_million: float | None       # USD per million input tokens; None when not stated
    completion_per_million: float | None
    free: bool

    def label(self) -> str:
        ctx = f"{self.context_length // 1000}k" if self.context_length else "?"
        price = "free" if self.free else (f"${self.prompt_per_million:g}/M in" if self.prompt_per_million is not None else "price ?")
        return f"{self.name} | {ctx} | {price}"


def _per_million(value: Any) -> float | None:
    try:
        return round(float(value) * 1_000_000, 6)
    except (TypeError, ValueError):
        return None


def parse_catalogue(payload: Any) -> list[ModelInfo]:
    """OpenRouter's model list -> ModelInfo. Tolerant: an entry without an id is skipped, a missing or odd field becomes None."""
    out: list[ModelInfo] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    for m in data if isinstance(data, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        pricing = m.get("pricing") or {}
        prompt, completion = _per_million(pricing.get("prompt")), _per_million(pricing.get("completion"))
        ctx = m.get("context_length")
        out.append(ModelInfo(id=str(m["id"]), name=str(m.get("name") or m["id"]), context_length=int(ctx) if isinstance(ctx, (int, float)) else None,
                             prompt_per_million=prompt, completion_per_million=completion,
                             free=str(m["id"]).endswith(":free") or (prompt == 0.0 and completion == 0.0)))
    return sorted(out, key=lambda x: x.id)


def _fetch_json(url: str, timeout: float = 10.0) -> Any:
    import requests
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


_cache: dict = {"at": 0.0, "models": []}
_cache_lock = threading.Lock()


def fetch_models(fetch: Callable[[str], Any] | None = None, force: bool = False) -> list[ModelInfo]:
    """OpenRouter's current models, from its public list (no key). Kept for an hour. Raises if it cannot be read."""
    with _cache_lock:
        if not force and _cache["models"] and time.time() - _cache["at"] < CATALOGUE_TTL:
            return list(_cache["models"])
    models = parse_catalogue((fetch or _fetch_json)(CATALOGUE_URL))
    if not models:
        raise RuntimeError("OpenRouter's model list came back empty or in a format this version does not understand")
    with _cache_lock:
        _cache.update(at=time.time(), models=models)
    return list(models)


def search_models(models: list[ModelInfo], text: str = "", free_only: bool = False, limit: int = 50) -> list[ModelInfo]:
    """Models whose id or name contains every word of `text` (any case), free ones only if asked, at most `limit`."""
    words = [w for w in text.lower().split() if w]
    hits = [m for m in models if (not free_only or m.free) and all(w in f"{m.id} {m.name}".lower() for w in words)]
    return hits[:limit] if limit else hits


class OpenRouterWrapper(LiteLLMWrapper):
    name = "openrouter"

    def __init__(self, config):
        super().__init__(config)
        self.app_title = str(config.extra.get("app_title", APP_TITLE))
        self.app_url = str(config.extra.get("app_url", ""))

    def _kwargs(self, syscall) -> dict:
        kw = super()._kwargs(syscall)
        headers = {"X-Title": self.app_title}
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        kw["extra_headers"] = {**headers, **kw.get("extra_headers", {})}
        return kw

    @staticmethod
    def models(fetch: Callable[[str], Any] | None = None, force: bool = False) -> list[ModelInfo]:
        return fetch_models(fetch, force)
