"""LiteLLM is the one wrapper for every cloud API. These tests use a fake `litellm` module (always run); the last section runs
against the REAL pinned library, offline, and skips itself when it is not installed."""
import os
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
from tests.fakes import FakeLiteLLM

from pal.errors import DependencyMissingError
from pal.llm_wrapper import (
    DeploymentType,
    LLMAdapter,
    LLMAdapterSyscall,
    LLMSyscall,
    WrapperConfig,
    build_wrapper,
)
from pal.llm_wrapper.components import litellm_wrapper
from pal.llm_wrapper.components.litellm_wrapper import PINNED, LiteLLMWrapper, full_model

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "weather in Pune?"}]
PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "weather", "parameters": PARAMS}}


def core(fake, backend="anthropic", model="claude-x", **kw):
    fake.__class__  # the fixture already installed it
    return build_wrapper(WrapperConfig(backend=backend, model=model, api_key="k-secret-123", **kw))


# -- model names ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend,model,want", [
    ("openai", "gpt-4o", "openai/gpt-4o"), ("anthropic", "claude-x", "anthropic/claude-x"),
    ("gemini", "gemini-3.5-flash", "gemini/gemini-3.5-flash"), ("groq", "llama-3.3-70b", "groq/llama-3.3-70b"),
    ("together", "meta/llama", "together_ai/meta/llama"), ("azure_openai", "my-deployment", "azure/my-deployment"),
    ("bedrock", "anthropic.claude-x", "bedrock/anthropic.claude-x"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", "openrouter/meta-llama/llama-3.3-70b-instruct:free"),
    ("openai", "openai/gpt-4o", "openai/gpt-4o"),                                   # never prefixed twice
    ("litellm", "cohere/command-r", "cohere/command-r")])                          # the raw backend takes the string as given
def test_model_strings_get_the_right_provider_prefix(backend, model, want):
    assert full_model(backend, model) == want


def test_every_cloud_provider_name_builds_a_litellm_wrapper(fake_litellm):
    for name in ("openai", "azure_openai", "anthropic", "gemini", "groq", "together", "mistral", "deepseek", "openrouter",
                 "perplexity", "bedrock", "google"):
        assert isinstance(build_wrapper(WrapperConfig(backend=name, model="m", api_key="k")), LiteLLMWrapper)


# -- the request -----------------------------------------------------------------------------------------------------------
def test_the_request_carries_the_key_per_call_never_in_the_environment(fake_litellm):
    before = dict(os.environ)
    c = core(fake_litellm, base_url="https://gateway.example/v1")
    c.chat(LLMSyscall(messages=MSGS, temperature=0.2, max_tokens=50))
    kw = fake_litellm.calls[0]
    assert kw["model"] == "anthropic/claude-x" and kw["api_key"] == "k-secret-123" and kw["api_base"] == "https://gateway.example/v1"
    assert (kw["temperature"], kw["max_tokens"], kw["num_retries"], kw["drop_params"]) == (0.2, 50, 0, True)
    assert kw["messages"] == MSGS and "tools" not in kw and "response_format" not in kw
    assert dict(os.environ) == before and "k-secret-123" not in " ".join(os.environ.values())


def test_tools_and_structured_output_are_sent_in_the_openai_shape(fake_litellm):
    from pal.llm_wrapper import LLMAdapterSyscall
    c = core(fake_litellm)
    c.chat(LLMAdapterSyscall(messages=MSGS, tools=[{"name": "get_weather", "description": "weather", "parameters": PARAMS}],
                             response_format=PARAMS))
    kw = fake_litellm.calls[0]
    assert kw["tools"] == [TOOL]
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {"name": "response", "schema": PARAMS}}
    c.chat(LLMAdapterSyscall(messages=MSGS, response_format={"type": "json_object"}))
    assert fake_litellm.calls[1]["response_format"] == {"type": "json_object"}


def test_only_safe_extra_parameters_pass_through_a_caller_cannot_redirect_the_key(fake_litellm):
    c = core(fake_litellm)
    c.chat(LLMSyscall(messages=MSGS, params={"top_p": 0.5, "stop": ["x"], "api_base": "https://evil.example", "api_key": "stolen",
                                              "model": "other"}))
    kw = fake_litellm.calls[0]
    assert kw["top_p"] == 0.5 and kw["stop"] == ["x"]
    assert kw["api_key"] == "k-secret-123" and "api_base" not in kw and kw["model"] == "anthropic/claude-x"
    c2 = build_wrapper(WrapperConfig(backend="azure_openai", model="d", api_key="k", extra={"litellm_params": {"api_version": "2025-01-01"}}))
    c2.chat(LLMSyscall(messages=MSGS))
    assert fake_litellm.calls[-1]["api_version"] == "2025-01-01"


# -- the reply -----------------------------------------------------------------------------------------------------------------
def test_reply_text_tokens_cost_and_tool_calls_are_unified(fake_litellm):
    fake_litellm.reply, fake_litellm.usage = "hello", (12, 4)
    fake_litellm.tool_calls = [("get_weather", '{"city": "Pune"}'), ("broken", "not json")]
    r = core(fake_litellm).chat(LLMSyscall(messages=MSGS))
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("hello", 12, 4)
    assert r.meta["tool_calls"] == [{"name": "get_weather", "parameters": {"city": "Pune"}}, {"name": "broken", "parameters": {"_raw": "not json"}}]
    assert r.meta["cost_usd"] == 0.0001 and r.meta["model"] == "anthropic/claude-x"
    fake_litellm.reply, fake_litellm.tool_calls, fake_litellm.cost = None, None, None
    r2 = core(fake_litellm).chat(LLMSyscall(messages=MSGS))
    assert r2.text == "" and r2.meta["tool_calls"] is None and r2.meta["cost_usd"] is None      # no price known: None, not a crash


def test_errors_from_the_provider_propagate_as_they_are(fake_litellm):
    fake_litellm.error = ConnectionError("provider down")
    with pytest.raises(ConnectionError, match="provider down"):
        core(fake_litellm).chat(LLMSyscall(messages=MSGS))


def test_streaming_yields_text_pieces_and_closes_the_stream(fake_litellm):
    assert list(core(fake_litellm).stream_chat(LLMSyscall(messages=MSGS))) == ["str", "eam"]
    assert fake_litellm.calls[0]["stream"] is True and fake_litellm.closed == 1
    gen = core(fake_litellm).stream_chat(LLMSyscall(messages=MSGS))
    next(gen)
    gen.close()                                                                       # stopped early: still closed
    assert fake_litellm.closed == 2


# -- facts about the model -------------------------------------------------------------------------------------------------------
def test_token_counts_prices_and_capabilities_come_from_litellm_with_safe_fallbacks(fake_litellm):
    c = core(fake_litellm)
    assert c.token_count("a b c") == 3 and c.price_per_token() == (3e-6, 15e-6)
    caps = c.capabilities()
    assert (caps.deployment_type, caps.supports_tools, caps.supports_json, caps.billed_per_token) == (DeploymentType.CLOUD_API, True, True, True)
    other = core(fake_litellm, model="mystery")
    assert other.price_per_token() is None
    fake_litellm.model_cost["gpt-4o"] = {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 1e-5}
    assert build_wrapper(WrapperConfig(backend="openai", model="gpt-4o", api_key="k")).price_per_token() is None   # "openai/gpt-4o" is not in this fake
    assert build_wrapper(WrapperConfig(backend="openrouter", model="gpt-4o", api_key="k")).price_per_token() is None   # and never borrows the bare name

    class Broken(FakeLiteLLM):
        def token_counter(self, **kw):
            raise RuntimeError("no tokenizer")

        def supports_function_calling(self, **kw):
            raise RuntimeError("unknown")

        def get_supported_openai_params(self, **kw):
            raise RuntimeError("unknown")

    litellm_wrapper._module = Broken()
    d = core(litellm_wrapper._module)
    assert d.token_count("one two three four") >= 1 and d.capabilities().supports_tools is False


# -- the adapter on top of it -----------------------------------------------------------------------------------------------------------
def test_the_adapter_runs_tool_calls_and_json_on_litellm_and_maps_its_errors(fake_litellm):
    fake_litellm.tool_calls = [("get_weather", '{"city": "Pune"}')]
    a = LLMAdapter([{"name": "claude-x", "backend": "anthropic", "api_key": "k-secret-123"}])
    r = a.llm_chat_with_tool_call_output(MSGS, [TOOL])
    assert r.tool_calls == [{"name": "get_weather", "parameters": {"city": "Pune"}}] and (r.prompt_tokens, r.completion_tokens) == (10, 3)
    fake_litellm.error = type("RateLimitError", (Exception,), {"status_code": 429})("429 slow down for key k-secret-123")
    bad = a.llm_chat(MSGS)
    assert bad.status_code == 429 and "k-secret-123" not in bad.error and "***" in bad.error


def test_a_missing_install_gives_a_clear_pinned_instruction(monkeypatch):
    monkeypatch.setattr(litellm_wrapper, "_module", None)
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(DependencyMissingError, match=re.escape(f"litellm=={PINNED}")):
        build_wrapper(WrapperConfig(backend="openai", model="m", api_key="k"))


def test_the_pin_the_extra_and_the_hash_lock_all_agree():
    root = Path(__file__).resolve().parents[2]
    lock = (root / "requirements" / "litellm.lock").read_text(encoding="utf-8")
    assert f"litellm=={PINNED} \\" in lock and lock.count("--hash=sha256:") > 100
    assert f"litellm=={PINNED}" in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert (root / "requirements" / "litellm.in").read_text(encoding="utf-8").strip() == f"litellm=={PINNED}"


# -- the REAL pinned library, offline (skips when it is not installed) --------------------------------------------------------------------
@pytest.fixture
def real():
    pytest.importorskip("litellm")
    litellm_wrapper._module = None                     # make sure the real one is imported through our safe wrapper
    yield litellm_wrapper._import_litellm()
    litellm_wrapper._module = None


def test_real_litellm_is_the_pinned_version_and_our_safety_settings_are_on(real):
    assert metadata.version("litellm") == PINNED
    assert real.suppress_debug_info is True and real.success_callback == [] and os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_real_litellm_chat_stream_errors_and_prices_work_without_a_network(real):
    c = build_wrapper(WrapperConfig(backend="anthropic", model="claude-sonnet-4-5", api_key="k-secret-123",
                              extra={"litellm_params": {"mock_response": "hello from mock"}}))
    r = c.chat(LLMSyscall(messages=MSGS))
    assert r.text == "hello from mock" and r.prompt_tokens > 0 and r.completion_tokens > 0
    assert "".join(c.stream_chat(LLMSyscall(messages=MSGS))) == "hello from mock"
    assert c.token_count("hello world how are you") >= 4
    err = build_wrapper(WrapperConfig(backend="gemini", model="gemini-2.5-flash", api_key="k", extra={"litellm_params": {
        "mock_response": real.RateLimitError("slow down", llm_provider="gemini", model="gemini-2.5-flash")}}))
    with pytest.raises(real.RateLimitError) as e:
        err.chat(LLMSyscall(messages=MSGS))
    assert e.value.status_code == 429
    a = LLMAdapter([{"name": "gemini-2.5-flash", "backend": "gemini", "api_key": "k", "litellm_params": {}}])
    assert a.llms[0].core.model == "gemini/gemini-2.5-flash"


def test_real_litellm_knows_prices_and_function_calling_for_a_common_model(real):
    c = build_wrapper(WrapperConfig(backend="openai", model="gpt-4o", api_key="k"))
    price = c.price_per_token()
    assert price and price[0] > 0 and price[1] > price[0]
    caps = c.capabilities()
    assert caps.supports_tools is True and caps.supports_json is True


def test_adapter_syscall_class_is_used_as_is(fake_litellm):
    assert LLMAdapterSyscall(messages=MSGS).action_type == "chat"
