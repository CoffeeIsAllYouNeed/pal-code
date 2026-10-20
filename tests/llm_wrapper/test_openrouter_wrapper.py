"""The OpenRouter wrapper: one key, hundreds of models, and a model browser. The model list is read through an injected fetch: no network here."""
import time

import pytest

from pal.llm_wrapper import LLMSyscall, WrapperConfig, build_wrapper
from pal.llm_wrapper.components import openrouter_wrapper as orw
from pal.llm_wrapper.components.litellm_wrapper import LiteLLMWrapper
from pal.llm_wrapper.components.openrouter_wrapper import (
    ModelInfo,
    OpenRouterWrapper,
    fetch_models,
    parse_catalogue,
    search_models,
)

PAYLOAD = {"data": [
    {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128000, "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta: Llama 3.3 70B (free)", "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "acme/zero-priced", "name": "Acme Zero", "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "anthropic/claude-x", "name": "Claude X", "pricing": {}},
    {"name": "no id, skipped"}, "not a dict", {"id": "odd/fields", "name": None, "context_length": "lots", "pricing": {"prompt": "n/a", "completion": None}}]}


@pytest.fixture(autouse=True)
def fresh_cache():
    orw._cache.update(at=0.0, models=[])
    yield
    orw._cache.update(at=0.0, models=[])


def test_the_model_list_is_read_into_names_context_sizes_prices_per_million_and_a_free_flag():
    models = {m.id: m for m in parse_catalogue(PAYLOAD)}
    assert sorted(models) == ["acme/zero-priced", "anthropic/claude-x", "meta-llama/llama-3.3-70b-instruct:free", "odd/fields", "openai/gpt-4o"]
    gpt = models["openai/gpt-4o"]
    assert (gpt.name, gpt.context_length, gpt.prompt_per_million, gpt.completion_per_million, gpt.free) == ("OpenAI: GPT-4o", 128000, 2.5, 10.0, False)
    assert models["meta-llama/llama-3.3-70b-instruct:free"].free and models["acme/zero-priced"].free               # free by the :free tag, or by a zero price
    assert models["anthropic/claude-x"].prompt_per_million is None and not models["anthropic/claude-x"].free         # unknown price is not "free"
    odd = models["odd/fields"]
    assert odd.name == "odd/fields" and odd.context_length is None and odd.prompt_per_million is None and not odd.free  # odd fields become None, nothing raises


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"data": None}, {"data": "nope"}])
def test_a_list_in_an_unexpected_shape_gives_no_models_not_an_error(payload):
    assert parse_catalogue(payload) == []


def test_a_model_describes_itself_in_one_line():
    assert ModelInfo("a/b", "A B", 128000, 2.5, 10.0, False).label() == "A B | 128k | $2.5/M in"
    assert ModelInfo("a/b:free", "A B", 8000, 0.0, 0.0, True).label() == "A B | 8k | free"
    assert ModelInfo("a/b", "A B", None, None, None, False).label() == "A B | ? | price ?"


def test_the_list_is_fetched_once_kept_for_an_hour_then_fetched_again(monkeypatch):
    calls = []
    fetch = lambda url: calls.append(url) or PAYLOAD  # noqa: E731
    first = fetch_models(fetch)
    assert calls == [orw.CATALOGUE_URL] == ["https://openrouter.ai/api/v1/models"] and len(first) == 5
    fetch_models(fetch)
    assert len(calls) == 1                                                                   # served from the cache
    fetch_models(fetch, force=True)
    assert len(calls) == 2
    monkeypatch.setattr(time, "time", lambda: orw._cache["at"] + orw.CATALOGUE_TTL + 1)
    fetch_models(fetch)
    assert len(calls) == 3                                                                   # an hour later


def test_an_empty_or_unreadable_list_is_an_error_never_a_stale_guess():
    with pytest.raises(RuntimeError, match="empty or in a format"):
        fetch_models(lambda url: {"data": []})
    def boom(url):
        raise OSError("no network")
    with pytest.raises(OSError):
        fetch_models(boom)
    assert orw._cache["models"] == []                                                        # a failure caches nothing


def test_searching_needs_every_word_in_the_id_or_name_free_only_and_a_limit():
    models = parse_catalogue(PAYLOAD)
    assert [m.id for m in search_models(models, "llama 70b")] == ["meta-llama/llama-3.3-70b-instruct:free"]
    assert [m.id for m in search_models(models, "OPENAI")] == ["openai/gpt-4o"] and search_models(models, "nothing here") == []
    assert {m.id for m in search_models(models, free_only=True)} == {"meta-llama/llama-3.3-70b-instruct:free", "acme/zero-priced"}
    assert len(search_models(models, limit=2)) == 2 and len(search_models(models, limit=0)) == 5


def test_the_registry_builds_the_openrouter_wrapper_on_top_of_the_litellm_wrapper(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="openrouter", model="meta-llama/llama-3.3-70b-instruct:free", api_key="or-key-123"))
    assert isinstance(w, OpenRouterWrapper) and isinstance(w, LiteLLMWrapper) and not w.local
    assert w.model == "openrouter/meta-llama/llama-3.3-70b-instruct:free"


def test_a_request_carries_the_key_per_call_and_the_apps_name_as_attribution(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="openrouter", model="openai/gpt-4o", api_key="or-key-123"))
    w.chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    kw = fake_litellm.calls[0]
    assert kw["api_key"] == "or-key-123" and kw["extra_headers"] == {"X-Title": "Pal"} and kw["model"] == "openrouter/openai/gpt-4o"


def test_the_app_title_and_url_can_be_set_and_an_explicit_header_wins(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="openrouter", model="m", api_key="k", extra={"app_title": "My App", "app_url": "https://my.app",
                                                                                          "litellm_params": {"extra_headers": {"X-Title": "Override", "X-Other": "1"}}}))
    w.chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    assert fake_litellm.calls[0]["extra_headers"] == {"X-Title": "Override", "HTTP-Referer": "https://my.app", "X-Other": "1"}


def test_the_wrapper_reads_the_list_through_the_same_function_and_a_plain_provider_has_no_attribution_header(fake_litellm):
    assert [m.id for m in OpenRouterWrapper.models(lambda url: PAYLOAD)][:1] == ["acme/zero-priced"]
    w = build_wrapper(WrapperConfig(backend="anthropic", model="claude-x", api_key="k"))
    w.chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    assert "extra_headers" not in fake_litellm.calls[-1]
