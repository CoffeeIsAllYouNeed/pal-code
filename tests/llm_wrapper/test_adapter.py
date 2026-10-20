import time

import pytest
import requests

from pal.errors import ConfigError, ProviderError
from pal.llm_wrapper import (
    DeploymentType,
    LLMAdapter,
    LLMAdapterSyscall,
    LLMConfig,
    LLMResponse,
    LLMSyscall,
    LLMWrapper,
    RouterStrategy,
    WrapperCapabilities,
)
from pal.llm_wrapper.utils.registry import WRAPPER_REGISTRY

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


class ScriptedCore(LLMWrapper):
    """A fake LLM instance. `extra`: replies (list of str or dict with text/tool_calls), tools, json (native flags),
    chunks (for streaming), boom (an exception to raise)."""

    name = "scripted"

    def __init__(self, config):
        super().__init__(config)
        x = config.extra
        self.replies = list(x.get("replies", ["ok"]))
        self.native_tools, self.native_json = x.get("tools", False), x.get("json", False)
        self.chunks, self.boom = x.get("chunks"), x.get("boom")
        self.seen: list[LLMSyscall] = []

    def chat(self, syscall):
        self.seen.append(syscall)
        if self.boom:
            raise self.boom
        r = self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]
        text, calls = (r, None) if isinstance(r, str) else (r.get("text", ""), r.get("tool_calls"))
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=4, meta={"tool_calls": calls})

    def stream_chat(self, syscall):
        self.seen.append(syscall)
        last = syscall.messages[-1]["content"]
        if last.startswith("[Continue"):
            done = syscall.messages[-2]["content"]
            todo = "".join(self.chunks)[len(done):]
            yield from [c for c in self.chunks if c in todo]
            return
        for c in self.chunks:
            time.sleep(0.05)
            yield c

    def token_count(self, text):
        return len(text.split())

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER, supports_tools=self.native_tools,
                                supports_json=self.native_json)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setitem(WRAPPER_REGISTRY, "fake", ScriptedCore)


def adapter(*cfgs, **kw):
    return LLMAdapter([{"name": f"m{i}", "backend": "fake", **c} for i, c in enumerate(cfgs or [{}])], **kw)


def core(a, i=0):
    return a.llms[i].core


MSG = [{"role": "user", "content": "hello"}]


# -- setup -------------------------------------------------------------------------------------------------------
def test_adapter_needs_at_least_one_llm_and_reports_bad_backends():
    with pytest.raises(ConfigError, match="at least one"):
        LLMAdapter([])
    with pytest.raises(ConfigError, match="nope/x"):
        LLMAdapter([{"name": "x", "backend": "nope"}])


def test_one_core_per_configured_instance_with_the_paper_style_configs():
    a = adapter({"extra_flag": 1}, {})
    assert [x.name for x in a.llms] == ["m0", "m1"] and a.strategy is RouterStrategy.Sequential
    assert a.llm_configs[0].extra == {"extra_flag": 1}


def test_api_keys_config_wins_then_given_key_then_environment(monkeypatch, fake_litellm):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")
    a = LLMAdapter([{"name": "m", "backend": "openai", "api_key": "explicit"}, {"name": "m", "backend": "openai"},
                    {"name": "q", "backend": "anthropic"}, {"name": "l", "backend": "ollama"}], api_key=["", "from-list", "ant"])
    assert [c.api_key for c in a.llm_configs] == ["explicit", "from-list", "ant", None]
    b = LLMAdapter([{"name": "m", "backend": "openai"}, {"name": "l", "backend": "ollama"}], api_key="shared")
    assert [c.api_key for c in b.llm_configs] == ["shared", None]            # a local runtime never takes a key
    c = LLMAdapter([{"name": "m", "backend": "openai"}])
    assert c.llm_configs[0].api_key == "env-openai" and c.llms[0].core.api_key == "env-openai"


def test_paper_backend_names_resolve_to_ours(fake_litellm):
    a = LLMAdapter([{"name": "x", "backend": "vllm", "hostname": "http://localhost:8000/v1"},
                    {"name": "gemini-x", "backend": "google", "api_key": "k"}])
    assert type(a.llms[0].core).__name__ == "LocalLLM" and a.llms[0].core.model == "hosted_vllm/x"   # a local runtime: the Local LLM component (LiteLLM underneath)
    assert type(a.llms[1].core).__name__ == "LiteLLMWrapper" and a.llms[1].core.model == "gemini/gemini-x"   # Google: through LiteLLM
    from pal.llm_wrapper.utils.registry import BACKEND_ALIASES
    assert BACKEND_ALIASES["google"] == "gemini" and BACKEND_ALIASES["huggingface"] == "huggingface_local"


# -- routing ------------------------------------------------------------------------------------------------------
def test_sequential_strategy_takes_the_instances_in_turn():
    a = adapter({"replies": ["from a"]}, {"replies": ["from b"]})
    assert [a.llm_chat(MSG).response_message for _ in range(4)] == ["from a", "from b", "from a", "from b"]
    assert [r.model for r in (a.llm_chat(MSG), a.llm_chat(MSG))] == ["m0", "m1"]


def test_a_call_can_name_the_instances_it_may_use():
    a = adapter({"replies": ["from a"]}, {"replies": ["from b"]}, {"replies": ["from c"]})
    assert [a.llm_chat(MSG, llms=["m1"]).response_message for _ in range(2)] == ["from b", "from b"]
    assert [a.llm_chat(MSG, llms=["m0", "m2"]).response_message for _ in range(3)] == ["from a", "from c", "from a"]
    assert a.llm_chat(MSG, llms=["fake/m2"]).response_message == "from c"
    bad = a.llm_chat(MSG, llms=["ghost"])
    assert bad.error and bad.status_code == 400 and "no configured LLM" in bad.error


# -- plain chat and the unified response -----------------------------------------------------------------------------
def test_chat_returns_the_unified_response_with_tokens_model_and_timing():
    a = adapter({"replies": ["hi there"]})
    r = a.llm_chat(MSG, max_tokens=33, temperature=0.3)
    assert (r.response_message, r.finished, r.error, r.status_code, r.tool_calls) == ("hi there", True, None, 200, None)
    assert (r.prompt_tokens, r.completion_tokens, r.model, r.backend) == (10, 4, "m0", "fake") and r.latency >= 0
    sent = core(a).seen[0]
    assert sent.max_tokens == 33 and sent.temperature == 0.3


def test_temperature_and_token_defaults_come_from_the_syscall_then_the_instance_then_the_argument():
    a = adapter({"temperature": 0.7, "max_new_tokens": 55})
    a.llm_chat(MSG)
    assert (core(a).seen[0].temperature, core(a).seen[0].max_tokens) == (0.7, 55)
    b = adapter({})
    b.execute_llm_syscall(LLMAdapterSyscall(messages=MSG), temperature=0.0)
    assert core(b).seen[0].temperature == 0.0
    b.execute_llm_syscall(LLMAdapterSyscall(messages=MSG, temperature=0.9))
    assert core(b).seen[1].temperature == 0.9


# -- structured output: native and emulated ---------------------------------------------------------------------------
def test_json_output_uses_the_native_mode_when_the_backend_has_it():
    a = adapter({"json": True, "replies": ['{"city": "Pune"}']})
    r = a.llm_chat_with_json_output(MSG, response_format={"type": "object", "properties": {"city": {}}})
    assert r.response_message == '{"city": "Pune"}' and r.error is None
    sent = core(a).seen[0]
    assert sent.response_format == {"type": "object", "properties": {"city": {}}} and sent.messages == MSG


def test_json_output_is_emulated_validated_and_retried_when_the_backend_lacks_it():
    a = adapter({"json": False, "replies": ["Sure, here you go: not json", '```json\n{"ok": true}\n```']})
    r = a.llm_chat_with_json_output(MSG)
    assert r.response_message == '{"ok": true}' and r.error is None
    first, retry = core(a).seen
    assert first.messages[0]["role"] == "system" and "valid JSON" in first.messages[0]["content"]
    assert "not valid JSON" in retry.messages[-1]["content"] and (r.prompt_tokens, r.completion_tokens) == (20, 8)


def test_json_that_never_validates_is_a_422_not_an_exception():
    a = adapter({"replies": ["nope", "still nope"]})
    r = a.llm_chat_with_json_output(MSG)
    assert r.response_message is None and r.status_code == 422 and "valid JSON" in r.error and r.finished


# -- function calling: native and emulated ------------------------------------------------------------------------------
def test_tool_calls_native_come_back_as_name_and_parameters():
    calls = [{"name": "get_weather", "parameters": {"city": "Pune"}}]
    a = adapter({"tools": True, "replies": [{"text": "", "tool_calls": calls}]})
    r = a.llm_chat_with_tool_call_output(MSG, [WEATHER])
    assert r.tool_calls == calls and r.response_message is None and r.error is None
    assert core(a).seen[0].tools == [{"name": "get_weather", "description": "weather", "parameters": WEATHER["function"]["parameters"]}]


def test_tool_calls_are_emulated_with_a_prompt_when_the_backend_lacks_them():
    reply = '{"tool_calls": [{"name": "get_weather", "parameters": {"city": "Pune"}}]}'
    a = adapter({"tools": False, "replies": [reply]})
    r = a.llm_chat_with_tool_call_output(MSG, [WEATHER])
    assert r.tool_calls == [{"name": "get_weather", "parameters": {"city": "Pune"}}]
    system = core(a).seen[0].messages[0]
    assert system["role"] == "system" and "get_weather" in system["content"] and getattr(core(a).seen[0], "tools", None) is None


def test_a_model_that_answers_in_text_instead_of_calling_a_tool_gets_that_text_back():
    a = adapter({"replies": ["It is sunny in Pune."]})
    r = a.llm_chat_with_tool_call_output(MSG, [WEATHER])
    assert r.tool_calls is None and r.response_message == "It is sunny in Pune." and r.error is None


def test_a_call_to_a_tool_that_was_not_offered_is_a_422():
    a = adapter({"tools": True, "replies": [{"tool_calls": [{"name": "launch_missiles", "parameters": {}}]}]})
    r = a.llm_chat_with_tool_call_output(MSG, [WEATHER])
    assert r.status_code == 422 and "unknown tool" in r.error and r.tool_calls is None


def test_a_tool_call_request_without_tools_and_unknown_actions_are_400s():
    a = adapter()
    assert a.llm_chat_with_tool_call_output(MSG, []).status_code == 400
    r = a.execute_llm_syscall(LLMAdapterSyscall(messages=MSG, action_type="chat_with_vision"))
    assert r.status_code == 400 and "unknown action_type" in r.error
    with pytest.raises(TypeError, match="unexpected argument"):
        a.llm_chat(MSG, colour="red")


# -- errors become responses -------------------------------------------------------------------------------------------
def http_error(code):
    e = requests.HTTPError(f"{code} error")
    e.response = type("R", (), {"status_code": code})()
    return e


@pytest.mark.parametrize("error,status", [
    (http_error(429), 429), (http_error(401), 401), (http_error(503), 503), (requests.Timeout("slow"), 504),
    (requests.ConnectionError("down"), 503), (ValueError("bad"), 400), (RuntimeError("boom"), 500),
    (RuntimeError("429 RESOURCE_EXHAUSTED quota"), 429), (RuntimeError("invalid API key"), 401)])
def test_failures_never_raise_and_map_to_a_status(error, status):
    a = adapter({"boom": error})
    r = a.llm_chat(MSG)
    assert r.error and r.status_code == status and r.response_message is None and r.finished


def test_keys_are_masked_in_error_messages():
    a = LLMAdapter([{"name": "m", "backend": "fake", "api_key": "super-secret-key-123"}])
    core(a).boom = RuntimeError("401 bad key super-secret-key-123 and sk-abcdefghijklmnop")
    r = a.llm_chat(MSG)
    assert "super-secret-key-123" not in r.error and "sk-abcdef" not in r.error and "***" in r.error


# -- interrupt and resume (use_context_manager) --------------------------------------------------------------------------
def test_a_long_generation_is_interrupted_at_the_time_limit_and_resumed_by_pid():
    a = adapter({"chunks": ["one ", "two ", "three ", "four "]}, use_context_manager=True)
    first = a.llm_chat(MSG, pid=7, time_limit=0.08)
    assert first.finished is False and first.response_message == "one two " and a.context_manager.load_context(7) is not None
    second = a.llm_chat(MSG, pid=7, time_limit=5)
    assert second.finished is True and second.response_message == "one two three four "
    assert a.context_manager.load_context(7) is None
    cont = core(a).seen[-1].messages
    assert cont[-2] == {"role": "assistant", "content": "one two "} and cont[-1]["content"].startswith("[Continue")


def test_interruption_is_off_without_the_flag_or_a_time_limit_and_never_applies_to_tools_or_json():
    a = adapter({"chunks": ["x"], "replies": ["whole reply"]})                       # no context manager
    assert a.llm_chat(MSG, pid=1, time_limit=0.01).response_message == "whole reply"
    b = adapter({"replies": ["whole reply"]}, use_context_manager=True)
    assert b.llm_chat(MSG).response_message == "whole reply"                          # no time limit
    assert b.llm_chat(MSG, pid=2, time_limit=0.01, max_tokens=5).finished in (True, False)
    j = b.llm_chat_with_json_output(MSG, pid=3, time_limit=0.01)
    assert j.error is not None and j.status_code == 422                               # the reply is not JSON; no interruption


# -- the adapter is itself a core ------------------------------------------------------------------------------------------
def test_the_adapter_drops_in_where_a_core_is_expected(tmp_path):
    a = adapter({"replies": ["via adapter"]})
    out = a.chat(LLMSyscall(messages=MSG, max_tokens=9))
    assert isinstance(out, LLMResponse) and out.text == "via adapter" and out.prompt_tokens == 10
    assert a.token_count("a b c") == 3 and a.capabilities().deployment_type is DeploymentType.LOCAL_SERVER
    from pal.sdk import AgentSDK
    sdk = AgentSDK.standalone(a, root=str(tmp_path / "s"))
    assert sdk.chat([{"role": "user", "content": "hi"}]) == "via adapter" and sdk.stats["calls"] == 1
    core(a).boom = RuntimeError("401 nope")
    with pytest.raises(ProviderError, match="nope"):
        a.chat(LLMSyscall(messages=MSG))


def test_concurrent_calls_get_distinct_usage_and_all_succeed():
    import threading
    a = adapter({"replies": ["x"]}, {"replies": ["y"]})
    out = []
    ts = [threading.Thread(target=lambda: out.append(a.llm_chat(MSG))) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(out) == 20 and all(r.ok and (r.prompt_tokens, r.completion_tokens) == (10, 4) for r in out)
    assert sorted(r.response_message for r in out).count("x") == 10


def test_llm_config_objects_work_as_well_as_dicts():
    a = LLMAdapter([LLMConfig(name="m", backend="fake", extra={"replies": ["obj"]})])
    assert a.llm_chat(MSG).response_message == "obj"
