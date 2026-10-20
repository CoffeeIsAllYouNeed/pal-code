"""The Local LLM component: models that run on this machine, behind a local server (Ollama, vLLM, LM Studio, llama.cpp).

It is the same LiteLLM wrapper underneath, so a local model has the same interface, token and error format as a cloud one, but it differs
where it matters: there is no key to protect (the OpenAI-style local servers get a placeholder, never a real key from the environment),
a model's first call may load it (a longer time-out), it costs nothing per token, and it can tell you which models it has installed.

The in-process Hugging Face model is the other local kind (huggingface_local.py): LiteLLM calls model servers and cannot load a model into
this process.
"""
from __future__ import annotations

from typing import Any

from pal.llm_wrapper.components.litellm_wrapper import LOCAL_PREFIX, LiteLLMWrapper

DEFAULT_URLS = {"ollama": "http://localhost:11434", "vllm": "http://localhost:8000/v1", "lmstudio": "http://localhost:1234/v1", "llamacpp": "http://localhost:8080/v1"}
LOCAL_BACKENDS = tuple(LOCAL_PREFIX)


class LocalLLM(LiteLLMWrapper):
    name = "local_llm"
    IS_LOCAL = True

    def __init__(self, config):
        if config.backend not in LOCAL_PREFIX:
            raise ValueError(f"{config.backend!r} is not a local runtime (local runtimes: {', '.join(LOCAL_BACKENDS)})")
        super().__init__(config)

    @property
    def url(self) -> str:
        return (self.config.base_url or self.config.host or DEFAULT_URLS[self.config.backend]).rstrip("/")

    def installed_models(self, timeout: float = 3.0, get: Any = None) -> list[str]:
        """The models this server has, as it names them: Ollama's /api/tags, or the OpenAI-style /models of the others. An empty list when the
        server cannot be reached (see `is_running`). `get` is for tests; it defaults to `requests.get`."""
        if get is None:
            import requests
            get = requests.get
        try:
            if self.config.backend == "ollama":
                r = get(f"{self.url}/api/tags", timeout=timeout)
                r.raise_for_status()
                return sorted(str(m.get("name") or m.get("model")) for m in r.json().get("models", []) if m.get("name") or m.get("model"))
            r = get(f"{self.url}/models", timeout=timeout)
            r.raise_for_status()
            return sorted(str(m["id"]) for m in r.json().get("data", []) if m.get("id"))
        except Exception:  # noqa: BLE001  (not running, not reachable, or not the API we expected)
            return []

    def is_running(self, timeout: float = 3.0, get: Any = None) -> bool:
        """Is the server up? (It answers a models request, even with no models yet.)"""
        if get is None:
            import requests
            get = requests.get
        try:
            path = "/api/tags" if self.config.backend == "ollama" else "/models"
            get(f"{self.url}{path}", timeout=timeout).raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False
