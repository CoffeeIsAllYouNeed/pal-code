"""Which model a session talks to: a local Ollama model, or an API model whose key is in the environment. Preferences are remembered between sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pal.errors import UserInputError
from pal.llm_wrapper.components.base import WrapperConfig
from pal.llm_wrapper.modes import PROVIDER_KEYS
from pal.llm_wrapper.utils.registry import build_wrapper
from pal.paths import pal_home
from pal.terminal.models import discover

LOCAL = "local"
DEFAULT_HOST = "http://localhost:11434"


def prefs_path() -> Path:
    return pal_home() / "pal.json"


def load_prefs() -> dict:
    try:
        data = json.loads(prefs_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        prov, model = data.get("provider", ""), data.get("model", "")
        if prov in PROVIDER_KEYS and model and split_spec(f"{prov}:{model}")[0] == LOCAL:     # an Ollama tag saved as an API model by mistake
            data["provider"], data["model"] = LOCAL, f"{prov}:{model}"
        return data
    except (OSError, ValueError):
        return {}


SAVED_PREFS = ("provider", "model", "mode", "helper_model", "setup_done", "model_ask", "model_plan", "model_build", "custom_base_url", "autonomy", "default_config")

# Beyond the named API providers and local Ollama:
#   litellm:<model>   any model LiteLLM knows (100+ providers: Azure, Bedrock, Vertex, xAI, Fireworks, Cohere ...); LiteLLM reads that provider's
#                     key from the environment under its usual name (XAI_API_KEY, AZURE_API_KEY ...)
#   custom:<model>    any OpenAI-compatible endpoint (a gateway, a hosted open model, LM Studio on another machine): the address comes from
#                     /endpoint (saved) or PAL_CUSTOM_BASE_URL, the key from PAL_CUSTOM_API_KEY (never saved; a placeholder when none, so a
#                     real OpenAI key in the environment is never sent to a third-party address)
#   lmstudio:<model>, vllm:<model>   local OpenAI-style servers (LM Studio on :1234, vLLM on :8000; PAL_LMSTUDIO_HOST / PAL_VLLM_HOST to change)
OPEN_PROVIDERS = ("litellm", "custom", "lmstudio", "vllm")
LOCAL_SERVERS = {"lmstudio": ("PAL_LMSTUDIO_HOST", "http://localhost:1234/v1"), "vllm": ("PAL_VLLM_HOST", "http://localhost:8000/v1")}


def known_provider(name: str) -> bool:
    return name == LOCAL or name in PROVIDER_KEYS or name in OPEN_PROVIDERS


def split_spec(spec: str) -> tuple[str, str]:
    """'provider:model' -> (provider, model); a bare name is a local Ollama model (whose own name may contain ':', like qwen3:8b)."""
    import re
    head, sep, rest = spec.strip().partition(":")
    if sep and head in PROVIDER_KEYS and re.fullmatch(r"(\d+(\.\d+)?[bmk]([-_.].*)?|latest|instruct.*|v\d.*|\d+x\d+b.*)", rest, re.I):
        return LOCAL, spec.strip()                          # an Ollama tag such as mistral:7b or deepseek:latest, not the API
    return (head, rest) if sep and known_provider(head) and head != LOCAL else (LOCAL, spec.strip().removeprefix("local:"))


def custom_base_url() -> str:
    return os.environ.get("PAL_CUSTOM_BASE_URL") or load_prefs().get("custom_base_url", "")


def save_prefs(**values) -> None:
    """Remember the last model and mode. Never a key: keys come from the environment only."""
    prefs = {**load_prefs(), **{k: v for k, v in values.items() if k in SAVED_PREFS and isinstance(v, str)}}
    prefs = {k: v for k, v in prefs.items() if v != "" or k in ("helper_model", "setup_done")}
    try:
        prefs_path().parent.mkdir(parents=True, exist_ok=True)
        prefs_path().write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        pass


def local_info(host: str = DEFAULT_HOST) -> dict:
    return discover(host)


def chat_models(info: dict) -> list[str]:
    return [m["name"] for m in info.get("models", []) if not m.get("embedding")]


def make_core(provider: str, model: str, host: str = DEFAULT_HOST, num_ctx: int = 8192):
    """The model wrapper for a provider ('local' or an API provider). Raises UserInputError with what to do when it cannot be made."""
    if provider == LOCAL:
        if not model:
            raise UserInputError("no local model is chosen: start Ollama and pull a model, or use an API model (pal --provider openrouter --model <name>)")
        return build_wrapper(WrapperConfig(backend="ollama", model=model, host=host, extra={"timeout": 900, "litellm_params": {"num_ctx": num_ctx, "keep_alive": "30m"}}))
    if not model:
        raise UserInputError(f"choose a model:  /model {provider}:<model name>")
    if provider == "litellm":
        return build_wrapper(WrapperConfig(backend="litellm", model=model, extra={"timeout": 300}))
    if provider == "custom":
        base = custom_base_url()
        if not base:
            raise UserInputError("set the endpoint first:  /endpoint https://your-server/v1   (or PAL_CUSTOM_BASE_URL)")
        return build_wrapper(WrapperConfig(backend="openai", model=model, base_url=base, api_key=os.environ.get("PAL_CUSTOM_API_KEY") or "not-needed",
                                           extra={"timeout": 300}))
    if provider in LOCAL_SERVERS:
        var, default = LOCAL_SERVERS[provider]
        return build_wrapper(WrapperConfig(backend=provider, model=model, host=os.environ.get(var, default), extra={"timeout": 900}))
    from pal.chat.providers import build_api_core
    if provider not in PROVIDER_KEYS:
        raise UserInputError(f"unknown provider {provider!r}: use local, one of {', '.join(PROVIDER_KEYS)}, or {', '.join(OPEN_PROVIDERS)}")
    if not os.environ.get(PROVIDER_KEYS[provider]):
        raise UserInputError(f"set {PROVIDER_KEYS[provider]} first, for example:  export {PROVIDER_KEYS[provider]}=...   (Windows: setx {PROVIDER_KEYS[provider]} ...)")
    return build_api_core(provider, model)


_HW = None


def hardware():
    """This computer, looked at once per run."""
    global _HW
    if _HW is None:
        from pal.terminal.hardware import detect
        _HW = detect()
    return _HW


def pick_default(info: dict, prefs: dict, hw=None) -> tuple[str, str]:
    """(provider, model) to start with: what was used last time if it still works, else the best installed local model for this GPU, else nothing."""
    provider, model = prefs.get("provider", ""), prefs.get("model", "")
    if provider == LOCAL and model in chat_models(info):
        return provider, model
    if provider in PROVIDER_KEYS and os.environ.get(PROVIDER_KEYS[provider]) and model:
        return provider, model
    if info.get("ok") and info.get("models"):
        from pal.terminal.local_models import best_installed
        best = best_installed(info["models"], hw or hardware(), exclude=prefs.get("helper_model", ""))
        if best:
            return LOCAL, best
    if info.get("ok") and info.get("recommended"):
        return LOCAL, info["recommended"]
    return LOCAL, ""


def make_helper(prefs: dict, info: dict, host: str = DEFAULT_HOST):
    """The small local model for Pal's own jobs, when setup chose one and it is installed; else None (the main model does those jobs)."""
    name = prefs.get("helper_model", "")
    if not name or name not in chat_models(info):
        return None
    try:
        return build_wrapper(WrapperConfig(backend="ollama", model=name, host=host, extra={"timeout": 300, "litellm_params": {"num_ctx": 4096, "keep_alive": "30m"}}))
    except Exception:  # noqa: BLE001  (litellm missing or misconfigured: fall back to the main model)
        return None
