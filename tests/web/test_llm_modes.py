"""The web side of the two modes: which modes there are, and OpenRouter's models for Explore."""
from tests.fakes import FakeAPI
from tests.helpers import client, make_service

PAYLOAD = {"data": [
    {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128000, "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta: Llama 3.3 70B", "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B", "context_length": 32000, "pricing": {"prompt": "0.0000002", "completion": "0.0000002"}}]}


def app(tmp_path, monkeypatch, fetch=lambda url: PAYLOAD, **env):
    from pal.llm_wrapper.components import openrouter_wrapper as orw
    orw._cache.update(at=0.0, models=[])
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    svc.catalogue_fetch = fetch
    return client(svc)


def test_the_page_is_told_both_modes_and_which_one_to_start_in(tmp_path, monkeypatch):
    body = app(tmp_path, monkeypatch, OPENROUTER_API_KEY="k").get("/api/llm/modes").get_json()
    assert body["default"] == "explore" and [m["key"] for m in body["modes"]] == ["explore", "own_key"]
    assert body["modes"][0]["ready"] and not body["modes"][1]["ready"]
    assert app(tmp_path / "b", monkeypatch, GEMINI_API_KEY="k").get("/api/llm/modes").get_json()["default"] == "own_key"


def test_the_config_the_page_loads_carries_the_modes_and_the_existing_provider_table(tmp_path, monkeypatch):
    cfg = app(tmp_path, monkeypatch, OPENAI_API_KEY="k").get("/api/config").get_json()
    assert cfg["llm_mode"] == "own_key" and {m["key"] for m in cfg["llm_modes"]} == {"explore", "own_key"}
    assert cfg["providers"]["openai"] is True and cfg["providers"]["openrouter"] is False         # unchanged


def test_explore_lists_openrouters_models_with_labels_and_a_free_flag(tmp_path, monkeypatch):
    models = app(tmp_path, monkeypatch).get("/api/llm/models").get_json()["models"]
    assert [m["id"] for m in models] == ["meta-llama/llama-3.3-70b-instruct:free", "mistralai/mistral-7b-instruct", "openai/gpt-4o"]
    assert models[0]["free"] is True and models[2]["free"] is False and models[2]["prompt_per_million"] == 2.5
    assert models[0]["label"] == "Meta: Llama 3.3 70B | 131k | free" and models[2]["label"] == "OpenAI: GPT-4o | 128k | $2.5/M in"


def test_the_list_can_be_searched_limited_and_narrowed_to_free_models(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch)
    assert [m["id"] for m in c.get("/api/llm/models?q=llama").get_json()["models"]] == ["meta-llama/llama-3.3-70b-instruct:free"]
    assert [m["id"] for m in c.get("/api/llm/models?free=1").get_json()["models"]] == ["meta-llama/llama-3.3-70b-instruct:free"]
    assert len(c.get("/api/llm/models?limit=2").get_json()["models"]) == 2 and c.get("/api/llm/models?q=zzz").get_json()["models"] == []


def test_a_bad_limit_is_refused_and_a_model_list_that_cannot_be_read_says_so_without_stopping_the_page(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch)
    r = c.get("/api/llm/models?limit=lots")
    assert r.status_code == 400 and "limit must be a number" in r.get_json()["error"]

    def down(url):
        raise OSError("no network")
    r = app(tmp_path / "d", monkeypatch, fetch=down).get("/api/llm/models")
    assert r.status_code == 502 and "could not read OpenRouter's model list" in r.get_json()["error"] and "type a model name yourself" in r.get_json()["error"]


def test_the_page_has_the_mode_switch_the_model_browser_and_the_new_stats_lines(tmp_path, monkeypatch):
    html = app(tmp_path, monkeypatch).get("/").get_data(as_text=True)
    for part in ('id="llmmode"', 'id="freeonly"', "/api/llm/models", "applyLlmMode", "where the time goes", "latency_stages", "RAM cache", "this process"):
        assert part in html, part
