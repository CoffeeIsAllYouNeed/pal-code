"""Fallback to a second model, dollars spent and saved (from LiteLLM's price table), and the spacing for OpenRouter free models."""
import json

import pytest
from tests.fakes import FakeAPI
from tests.helpers import client, make_service, send

from pal.config import load_settings
from pal.errors import ConfigError, ProviderError, UserInputError
from pal.llm_scheduler import LLMScheduler
from pal.llm_wrapper.components.base import LLMResponse

FREE = "meta-llama/llama-3.3-70b-instruct:free"
FB = {"enabled": True, "provider": "openrouter", "model": FREE}
FAST = {"api_backoff_seconds": 0.01, "api_retry_429": 1}


class Failing(FakeAPI):
    """An API that always raises `error`, counting the calls."""

    def __init__(self, error):
        super().__init__()
        self.error, self.calls = error, 0

    def chat(self, syscall):
        self.calls += 1
        raise self.error


def http_error(code, text="boom"):
    return type("HTTPError", (Exception,), {"status_code": code})(f"{code} {text}")


def two_models(tmp_path, primary, backup_api, **over):
    svc, _ = make_service(tmp_path, **over)
    apis = {"anthropic": primary, "openrouter": backup_api}
    svc.api_core = lambda provider, model, key=None: apis[provider]
    return svc


def test_fallback_is_off_by_default_so_a_rate_limit_is_still_reported(tmp_path):
    primary, backup = Failing(RuntimeError("429 quota")), FakeAPI(["backup"])
    svc = two_models(tmp_path, primary, backup, scheduler=FAST)
    with pytest.raises(ProviderError, match="rate limit"):
        send(svc, "explain the cache", route="api")
    assert backup.seen == [] and svc.metrics.chat_summary()["fallbacks"] == 0


def test_a_rate_limited_provider_is_retried_by_the_line_then_the_fallback_model_answers(tmp_path):
    primary, backup = Failing(RuntimeError("429 RESOURCE_EXHAUSTED")), FakeAPI(["answered by the backup"])
    svc = two_models(tmp_path, primary, backup, scheduler=FAST, fallback=FB)
    out = send(svc, "explain the cache", route="api")
    assert out.reply == "answered by the backup"
    assert out.fallback == {"provider": "openrouter", "model": FREE, "reason": "RuntimeError"}
    assert primary.calls == 2 and len(backup.seen) == 1                         # one call plus one retry, then the fallback
    assert svc.metrics.chat_summary()["fallbacks"] == 1 and svc.metrics.turns()[-1]["fallback"] == f"openrouter/{FREE}"


@pytest.mark.parametrize("error", [http_error(503), http_error(500), TimeoutError("slow"), ConnectionError("down")])
def test_server_errors_timeouts_and_connection_failures_also_use_the_fallback(tmp_path, error):
    svc = two_models(tmp_path, Failing(error), FakeAPI(["backup"]), scheduler=FAST, fallback=FB)
    assert send(svc, "explain the cache", route="api").reply == "backup"


@pytest.mark.parametrize("error", [http_error(401, "bad key"), http_error(400, "bad request"), http_error(404, "no such model"),
                                   ValueError("invalid")])
def test_a_bad_key_request_or_model_would_fail_anywhere_so_the_fallback_is_not_used(tmp_path, error):
    backup = FakeAPI(["backup"])
    svc = two_models(tmp_path, Failing(error), backup, scheduler=FAST, fallback=FB)
    with pytest.raises(ProviderError):
        send(svc, "explain the cache", route="api")
    assert backup.seen == []


def test_when_the_fallback_also_fails_the_original_problem_is_reported(tmp_path):
    svc = two_models(tmp_path, Failing(RuntimeError("429 quota for key k")), Failing(http_error(502, "gateway")), scheduler=FAST, fallback=FB)
    with pytest.raises(ProviderError, match="rate limit or quota"):
        send(svc, "explain the cache", route="api")


def test_a_missing_fallback_key_means_no_fallback_not_a_crash(tmp_path):
    svc, _ = make_service(tmp_path, scheduler=FAST, fallback=FB)
    primary = Failing(RuntimeError("429 quota"))

    def core(provider, model, key=None):
        if provider == "openrouter":
            raise UserInputError("no API key: paste one in settings or set OPENROUTER_API_KEY")
        return primary
    svc.api_core = core
    with pytest.raises(ProviderError, match="rate limit"):
        send(svc, "explain the cache", route="api")


def test_agents_use_the_fallback_unless_it_is_switched_off_for_them(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_CODING", "agent-key")
    for for_agents, expect_ok in ((True, True), (False, False)):
        svc = two_models(tmp_path / str(for_agents), Failing(RuntimeError("429 quota")), FakeAPI(["backup"]), scheduler=FAST,
                         fallback={**FB, "for_agents": for_agents})
        svc.api_core = (lambda p, m, k=None, svc=svc: Failing(RuntimeError("429 quota")) if p == "gemini" else FakeAPI(["backup"]))
        if expect_ok:
            assert svc.send("a", "", "", None, "explain the cache", [], agent="coding").reply == "backup"
        else:
            with pytest.raises(ProviderError):
                svc.send("a", "", "", None, "explain the cache", [], agent="coding")


def test_later_tool_rounds_stay_on_the_fallback_model(tmp_path):
    call = '<tool>{"name": "calculator", "args": {"expression": "17+25"}}</tool>'
    backup = FakeAPI([call, "the total is 42"])
    primary = Failing(RuntimeError("429 quota"))
    svc = two_models(tmp_path, primary, backup, scheduler=FAST, fallback=FB)
    out = send(svc, "how many is 17 plus 25 in total?", route="api")
    assert out.reply == "the total is 42" and out.tools and out.tools[0]["name"] == "calculator"
    assert len(backup.seen) == 2 and primary.calls == 2                         # the primary was not tried again in round two


# -- cost -------------------------------------------------------------------------------------------------------------------
class Priced(FakeAPI):
    def __init__(self, cost=0.002, price=(3e-6, 15e-6), replies=None):
        super().__init__(replies or ["ok"])
        self.cost, self.price = cost, price

    def chat(self, syscall):
        r = super().chat(syscall)
        return LLMResponse(text=r.text, prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, meta={"cost_usd": self.cost})

    def price_per_token(self):
        return self.price


def test_cost_is_taken_from_the_response_and_savings_are_priced_at_the_models_input_rate(tmp_path):
    svc, _ = make_service(tmp_path, api=Priced())
    out = send(svc, "please explain the caching layer", route="api")
    m = svc.metrics.turns()[-1]
    assert out.cost_usd == m["cost_usd"] == 0.002
    assert m["saved_cost_usd"] == pytest.approx((m["saved_savers"] + m["saved_paging"] + m["saved_graph"] + m["saved_documents"]) * 3e-6)
    assert m["saved_cost_usd"] > 0
    s = svc.metrics.chat_summary()
    assert s["cost_usd"] == 0.002 and s["saved_cost_usd"] == pytest.approx(m["saved_cost_usd"])


def test_tool_rounds_add_up_and_an_unknown_price_is_zero_not_an_error(tmp_path):
    call = '<tool>{"name": "calculator", "args": {"expression": "1+1"}}</tool>'
    svc, _ = make_service(tmp_path, api=Priced(replies=[call, "two"]))
    assert send(svc, "how many is 1 plus 1 please?", route="api").cost_usd == pytest.approx(0.004)
    svc2, _ = make_service(tmp_path / "b", api=Priced(cost=None, price=None))
    out = send(svc2, "please explain the caching layer", route="api")
    assert out.cost_usd == 0.0 and svc2.metrics.turns()[-1]["saved_cost_usd"] == 0.0


def test_the_web_response_carries_cost_and_fallback(tmp_path):
    svc = two_models(tmp_path, Failing(RuntimeError("429 quota")), Priced(replies=["backup"]), scheduler=FAST, fallback=FB)
    r = client(svc).post("/api/chat", data={"session_id": "w", "message": "explain the cache", "provider": "anthropic", "model": "m",
                                            "api_key": "k", "route": "api"}).get_json()
    assert r["fallback"]["model"] == FREE and r["cost_usd"] == 0.002 and r["reply"] == "backup"
    assert json.dumps(r).count("api_key") == 0


# -- free-model spacing and settings ---------------------------------------------------------------------------------------------------
def test_openrouter_free_models_are_spaced_other_models_follow_their_provider_setting():
    s = LLMScheduler({"free_model_interval": 3.0, "api_min_interval": {"gemini": 4.0}})
    assert s.interval_for("openrouter", FREE) == 3.0 and s.interval_for("openrouter", "anthropic/claude-x") == 0.0
    assert s.interval_for("gemini", "gemini-x") == 4.0 and s.interval_for("openai") == 0.0
    assert LLMScheduler({"free_model_interval": 1.5}).interval_for("openrouter", "x:free") == 1.5
    s.close()


def test_fallback_and_free_interval_settings_are_validated():
    assert load_settings(env={}).fallback.enabled is False and load_settings(env={}).scheduler.free_model_interval == 3.0
    s = load_settings(env={}, overrides={"fallback": {"enabled": True, "model": FREE}})
    assert (s.fallback.provider, s.fallback.model, s.fallback.for_agents) == ("openrouter", FREE, True)
    with pytest.raises(ConfigError, match=">= 0"):
        load_settings(env={}, overrides={"scheduler": {"free_model_interval": -1}})
    with pytest.raises(ConfigError, match="must not be empty"):
        load_settings(env={}, overrides={"fallback": {"provider": " "}})
