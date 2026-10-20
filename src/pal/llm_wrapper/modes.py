"""Two ways to reach a cloud model, and which providers belong to each.

  explore    One OpenRouter key reaches hundreds of models, some free: the way to try new models. Your prompts pass through OpenRouter.
  own_key    Your own key for one provider (Anthropic, OpenAI, Gemini ...): direct, no middleman, that provider's own terms.

Both are served by the same LiteLLM-based wrapper (the OpenRouter wrapper adds the model browser); a mode is only a choice of which providers to
offer. Local models are a third thing (the Local LLM component) and are not a mode.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# provider -> the environment variable that holds its key (the order is the order they are offered in)
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
EXPLORE_PROVIDER = "openrouter"


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    summary: str
    providers: tuple[str, ...]


MODES: dict[str, Mode] = {
    "explore": Mode("explore", "Explore models",
                    "One OpenRouter key reaches hundreds of models, some of them free. Your prompts pass through OpenRouter on the way to the model.",
                    (EXPLORE_PROVIDER,)),
    "own_key": Mode("own_key", "Use my own key",
                    "Your own key for one provider (Anthropic, OpenAI, Gemini ...). Direct to that provider, no middleman.",
                    tuple(p for p in PROVIDER_KEYS if p != EXPLORE_PROVIDER)),
}


def mode_for(provider: str) -> str:
    """Which mode a provider belongs to."""
    if provider not in PROVIDER_KEYS:
        raise KeyError(f"unknown provider {provider!r} (known: {', '.join(PROVIDER_KEYS)})")
    return "explore" if provider == EXPLORE_PROVIDER else "own_key"


def providers_with_keys(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Every provider and whether its key is set in the environment."""
    env = os.environ if env is None else env
    return {p: bool(env.get(var)) for p, var in PROVIDER_KEYS.items()}


def default_mode(env: Mapping[str, str] | None = None) -> str:
    """own_key if any provider key is set, else explore if only an OpenRouter key is, else own_key (the plain default)."""
    have = providers_with_keys(env)
    if any(v for p, v in have.items() if p != EXPLORE_PROVIDER):
        return "own_key"
    return "explore" if have[EXPLORE_PROVIDER] else "own_key"


def describe_modes(env: Mapping[str, str] | None = None) -> list[dict]:
    """The modes as plain data for the web page and the command line, with which providers already have a key."""
    have = providers_with_keys(env)
    return [{"key": m.key, "title": m.title, "summary": m.summary,
             "providers": [{"name": p, "key_env": PROVIDER_KEYS[p], "key_set": have[p]} for p in m.providers],
             "ready": any(have[p] for p in m.providers)} for m in MODES.values()]
