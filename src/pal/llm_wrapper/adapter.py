"""LLMAdapter: the LLM core, as AIOS section 3.2 and Appendix A.2 describe it.

Each LLM instance is wrapped as a *core* (like a CPU core in an operating system), whatever its deployment: a cloud API
(OpenAI, Anthropic, Google, Groq, Bedrock), a local server (vLLM, Ollama) or a model loaded in this process (Hugging Face).
The adapter holds one core per configured instance and answers unified inference system calls on them:

    llm_chat                        plain text
    llm_chat_with_json_output       structured (JSON) output
    llm_chat_with_tool_call_output  function calling

Every call returns one LLMAdapterResponse (response_message, tool_calls, finished, error, status_code). A failure is reported
in that response, never raised. With several instances the routing strategy picks one per request; with `use_context_manager`
a long generation can be interrupted at a time limit and resumed.

Function calling and structured output are native where the backend supports them and emulated (prompt plus validation)
where it does not; `WrapperCapabilities.supports_tools` and `supports_json` say which.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from pal.context_manager.components.simple_context_manager import SimpleContextManager
from pal.errors import ConfigError, ProviderError
from pal.llm_wrapper.components.adapter_types import (
    ACTIONS,
    LLMAdapterResponse,
    LLMAdapterSyscall,
    LLMConfig,
    RouterStrategy,
)
from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper, WrapperCapabilities
from pal.llm_wrapper.components.features import (
    ToolCallError,
    check_tool_names,
    extract_json,
    json_instruction,
    normalize_tools,
    parse_emulated_tool_calls,
    tools_instruction,
)
from pal.llm_wrapper.utils.registry import BACKEND_ALIASES, build_wrapper
from pal.redact import redact

log = logging.getLogger(__name__)

# Environment variables each cloud backend's key is read from when none is given.
_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",), "azure_openai": ("AZURE_OPENAI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "groq": ("GROQ_API_KEY",), "deepseek": ("DEEPSEEK_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",), "together": ("TOGETHER_API_KEY",), "openrouter": ("OPENROUTER_API_KEY",),
    "perplexity": ("PERPLEXITY_API_KEY",),
}
# Backends that never take an API key from us: local runtimes, and Bedrock (the AWS credential chain).
_KEYLESS = {"ollama", "vllm", "lmstudio", "llamacpp", "huggingface_local", "bedrock"}
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{10,}|AKIA[0-9A-Z]{12,}|Bearer\s+[A-Za-z0-9._\-]{12,})")


@dataclass
class _Llm:
    name: str
    core: LLMWrapper
    config: LLMConfig


class LLMAdapter:
    def __init__(self, llm_configs: list[dict | LLMConfig], api_key: str | list[str | None] | None = None,
                 log_mode: str = "console", use_context_manager: bool = False,
                 strategy: RouterStrategy = RouterStrategy.Sequential):
        if not llm_configs:
            raise ConfigError("the LLM adapter needs at least one LLM config")
        self.llm_configs = [c if isinstance(c, LLMConfig) else LLMConfig.from_dict(c) for c in llm_configs]
        self.api_key = api_key
        self.log_mode = log_mode
        self.strategy = RouterStrategy(strategy)
        self.context_manager = SimpleContextManager() if use_context_manager else None
        self.llms: list[_Llm] = []
        self._next = 0
        self._lock = threading.Lock()
        self._tls = threading.local()
        self.setup_api_keys()
        self.initialize_llms()

    # -- setup ------------------------------------------------------------------------------------------------
    def setup_api_keys(self) -> None:
        """A key given in the config wins; then `api_key` (one string for all, or a list matched by position); then the
        backend's usual environment variable. Local runtimes and Bedrock take no key from here."""
        for i, cfg in enumerate(self.llm_configs):
            if cfg.api_key:
                continue
            backend = BACKEND_ALIASES.get(cfg.backend, cfg.backend)
            given = (self.api_key[i] if i < len(self.api_key) else None) if isinstance(self.api_key, (list, tuple)) else self.api_key
            if backend in _KEYLESS:
                continue
            cfg.api_key = given or next((os.environ[v] for v in _ENV_KEYS.get(backend, ()) if os.environ.get(v)), None)

    def initialize_llms(self) -> None:
        for cfg in self.llm_configs:
            self.initialize_single_llm(cfg)

    def initialize_single_llm(self, config: LLMConfig) -> None:
        config.backend = BACKEND_ALIASES.get(config.backend, config.backend)
        try:
            core = build_wrapper(config.to_core_config())
        except Exception as e:
            raise ConfigError(f"could not set up {config.backend}/{config.name}: {self._mask(str(e))}") from e
        self.llms.append(_Llm(config.name, core, config))

    # -- unified system calls ----------------------------------------------------------------------------------
    def llm_chat(self, messages: list[dict], **kw: Any) -> LLMAdapterResponse:
        return self.execute_llm_syscall(LLMAdapterSyscall(messages=messages, action_type="chat", **self._kw(kw)))

    def llm_chat_with_json_output(self, messages: list[dict], response_format: dict | None = None,
                                  **kw: Any) -> LLMAdapterResponse:
        return self.execute_llm_syscall(LLMAdapterSyscall(
            messages=messages, action_type="chat_with_json_output", message_return_type="json",
            response_format=response_format, **self._kw(kw)))

    def llm_chat_with_tool_call_output(self, messages: list[dict], tools: list[dict], **kw: Any) -> LLMAdapterResponse:
        return self.execute_llm_syscall(LLMAdapterSyscall(
            messages=messages, action_type="chat_with_tool_call_output", tools=tools, **self._kw(kw)))

    @staticmethod
    def _kw(kw: dict) -> dict:
        allowed = {"llms", "temperature", "max_tokens", "pid", "time_limit", "model"}
        bad = set(kw) - allowed
        if bad:
            raise TypeError(f"unexpected argument(s): {', '.join(sorted(bad))}")
        return kw

    # -- the call --------------------------------------------------------------------------------------------------
    def execute_llm_syscall(self, llm_syscall: LLMAdapterSyscall, temperature: float = 0.0) -> LLMAdapterResponse:
        """Run one request on one LLM instance. `temperature` is the fallback when neither the syscall nor the instance
        sets one. Never raises: a failure comes back as an LLMAdapterResponse with `error` and `status_code`."""
        started = time.perf_counter()
        self._tls.usage = [0, 0]
        llm: _Llm | None = None
        try:
            if llm_syscall.action_type not in ACTIONS:
                raise ValueError(f"unknown action_type {llm_syscall.action_type!r}; use one of {', '.join(ACTIONS)}")
            llm = self._select(llm_syscall)
            temp = llm_syscall.temperature if llm_syscall.temperature is not None else (
                llm.config.temperature if llm.config.temperature is not None else temperature)
            max_tokens = llm_syscall.max_tokens or llm.config.max_new_tokens
            tools = llm_syscall.tools if llm_syscall.action_type == "chat_with_tool_call_output" else None
            if llm_syscall.action_type == "chat_with_tool_call_output" and not tools:
                raise ValueError("chat_with_tool_call_output needs at least one tool")
            mrt = "json" if llm_syscall.action_type == "chat_with_json_output" else llm_syscall.message_return_type
            if (self.context_manager and llm_syscall.time_limit and llm_syscall.pid is not None and not tools
                    and mrt == "text" and not llm_syscall.response_format):
                completed, finished = self.context_manager.generate_response_with_interruption(
                    llm.config.name, llm.core, llm_syscall.messages, tools, mrt, temp, max_tokens, llm_syscall.pid,
                    llm_syscall.time_limit, llm_syscall.response_format)
            else:
                completed = self.get_model_response(
                    llm.config.name, llm.core, llm_syscall.messages, tools, llm_syscall, llm.config.hostname, mrt,
                    llm_syscall.response_format, temp, max_tokens)
                finished = True
            resp = self.process_response(completed, finished, tools, llm.core, mrt)
        except Exception as e:  # noqa: BLE001  (every failure becomes a response)
            resp = self.handle_completion_error(e)
        resp.prompt_tokens, resp.completion_tokens = self._tls.usage
        resp.latency = round(time.perf_counter() - started, 4)
        if llm is not None:
            resp.model, resp.backend = llm.config.name, llm.config.backend
        if self.log_mode == "console":
            log.info("llm call backend=%s model=%s status=%s in=%d out=%d %.2fs", resp.backend or "-", resp.model or "-",
                     resp.status_code, resp.prompt_tokens, resp.completion_tokens, resp.latency)
        return resp

    def _select(self, syscall: LLMAdapterSyscall) -> _Llm:
        """RouterStrategy.Sequential: the instances in turn, one per request (among the named ones, if the call names some)."""
        with self._lock:
            pool = self.llms
            if syscall.llms:
                pool = [x for x in self.llms if x.name in syscall.llms or f"{x.config.backend}/{x.name}" in syscall.llms]
                if not pool:
                    raise ValueError(f"no configured LLM matches {syscall.llms}")
            chosen = pool[self._next % len(pool)]
            self._next += 1
            return chosen

    def get_model_response(self, model_name: str, model: LLMWrapper, messages: list[dict], tools: list[dict] | None,
                           llm_syscall: LLMAdapterSyscall, api_base: str | None = None, message_return_type: str = "text",
                           response_format: dict | None = None, temperature: float = 1.0, max_tokens: int = 1000) -> Any:
        """Ask the model. Returns text, a JSON string, or a list of tool calls: the 'completed response' that
        process_response turns into the unified response. `api_base` is part of the paper's signature; the core already
        holds its own endpoint."""
        caps: WrapperCapabilities = model.capabilities()

        def call(msgs: list[dict], **extra: Any) -> LLMResponse:
            sc = LLMAdapterSyscall(messages=msgs, model=model_name or None, temperature=temperature, max_tokens=max_tokens,
                                   **extra)
            r = model.chat(sc)
            self._tls.usage[0] += r.prompt_tokens or 0
            self._tls.usage[1] += r.completion_tokens or 0
            return r

        specs = normalize_tools(tools) if tools else []
        if specs:
            if caps.supports_tools:
                r = call(messages, tools=specs)
                calls = (r.meta or {}).get("tool_calls")
            else:
                r = call(self._with_instruction(messages, tools_instruction(specs)))
                calls = parse_emulated_tool_calls(r.text, specs)
            if calls:
                check_tool_names(calls, specs)
                return calls
            return r.text

        if message_return_type == "json" or response_format:
            schema = response_format if isinstance(response_format, dict) else None
            msgs = messages if caps.supports_json else self._with_instruction(messages, json_instruction(schema))
            r = call(msgs, response_format=(schema or {"type": "json_object"}) if caps.supports_json else None)
            try:
                extract_json(r.text)
            except ValueError as e:                       # one retry that tells the model what was wrong
                retry = msgs + [{"role": "assistant", "content": r.text},
                                {"role": "user", "content": f"That was not valid JSON ({e}). Reply again with only valid JSON."}]
                r = call(retry, response_format=(schema or {"type": "json_object"}) if caps.supports_json else None)
            return r.text
        return call(messages).text

    @staticmethod
    def _with_instruction(messages: list[dict], instruction: str) -> list[dict]:
        out = [dict(m) for m in messages]
        if out and out[0].get("role") == "system":
            out[0]["content"] = f"{out[0]['content']}\n\n{instruction}"
        else:
            out.insert(0, {"role": "system", "content": instruction})
        return out

    def process_response(self, completed_response: Any, finished: bool, tools: list[dict] | None = None,
                         model: Any = None, message_return_type: str | None = None) -> LLMAdapterResponse:
        if isinstance(completed_response, list):          # tool calls
            calls = [{"name": c["name"], "parameters": c.get("parameters") or {}} for c in completed_response]
            return LLMAdapterResponse(tool_calls=calls, finished=finished)
        text = completed_response if isinstance(completed_response, str) else str(completed_response or "")
        if message_return_type == "json":
            try:
                text = json.dumps(extract_json(text), ensure_ascii=False)
            except ValueError as e:
                return LLMAdapterResponse(error=f"the model did not return valid JSON: {e}", status_code=422, finished=True)
        return LLMAdapterResponse(response_message=text, finished=finished)

    def handle_completion_error(self, error: Exception) -> LLMAdapterResponse:
        return LLMAdapterResponse(response_message=None, finished=True, status_code=self._status_for(error),
                                  error=self._mask(f"{type(error).__name__}: {error}"))

    @staticmethod
    def _status_for(error: Exception) -> int:
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return int(error.response.status_code)
        for attr in ("status_code", "code"):
            code = getattr(error, attr, None)
            if isinstance(code, int) and 400 <= code <= 599:
                return code
        resp = getattr(error, "response", None)
        if isinstance(resp, dict) and isinstance(resp.get("ResponseMetadata", {}).get("HTTPStatusCode"), int):
            return resp["ResponseMetadata"]["HTTPStatusCode"]
        if isinstance(error, (requests.Timeout, TimeoutError)):
            return 504
        if isinstance(error, requests.ConnectionError):
            return 503
        if isinstance(error, ToolCallError):
            return 422
        low = str(error).lower()
        if "429" in low or "rate limit" in low or "quota" in low or "resource_exhausted" in low:
            return 429
        if "401" in low or "unauthorized" in low or "api key" in low:
            return 401
        if isinstance(error, (ValueError, TypeError, ConfigError)):
            return 400
        return 500

    def _mask(self, text: str) -> str:
        return redact(text, *(cfg.api_key for cfg in self.llm_configs))

    # -- drop-in as an LLMWrapper, so the SDK and the rest of Pal can run on the adapter ---------------------------
    @property
    def primary(self) -> LLMWrapper:
        return self.llms[0].core

    @property
    def config(self) -> Any:
        return self.primary.config

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        sc = syscall if isinstance(syscall, LLMAdapterSyscall) else LLMAdapterSyscall(
            messages=syscall.messages, model=syscall.model, temperature=syscall.temperature, max_tokens=syscall.max_tokens,
            stream=syscall.stream, params=syscall.params)
        resp = self.execute_llm_syscall(sc)
        if resp.error:
            raise ProviderError(resp.error)
        return resp.to_core_response()

    def stream_chat(self, syscall: LLMSyscall):
        return self.primary.stream_chat(syscall)

    def token_count(self, text: str) -> int:
        return self.primary.token_count(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.primary.embed(texts)

    def capabilities(self) -> WrapperCapabilities:
        return self.primary.capabilities()
