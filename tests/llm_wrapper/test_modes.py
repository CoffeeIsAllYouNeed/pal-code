"""The two ways to reach a cloud model: Explore (one OpenRouter key) and Own key (a key per provider)."""
import pytest

from pal.chat.providers import API_PROVIDERS
from pal.llm_wrapper.modes import (
    MODES,
    PROVIDER_KEYS,
    default_mode,
    describe_modes,
    mode_for,
    providers_with_keys,
)


def test_there_are_exactly_two_modes_and_between_them_they_cover_every_provider_once():
    assert list(MODES) == ["explore", "own_key"]
    assert MODES["explore"].providers == ("openrouter",)
    assert "openrouter" not in MODES["own_key"].providers
    both = list(MODES["explore"].providers) + list(MODES["own_key"].providers)
    assert sorted(both) == sorted(PROVIDER_KEYS) and len(both) == len(set(both))
    assert API_PROVIDERS is PROVIDER_KEYS                                          # the chat uses the same table


def test_every_provider_belongs_to_one_mode_and_an_unknown_one_is_an_error():
    assert mode_for("openrouter") == "explore" and mode_for("anthropic") == "own_key" and mode_for("gemini") == "own_key"
    with pytest.raises(KeyError, match="unknown provider 'cohere'"):
        mode_for("cohere")


def test_the_keys_found_in_the_environment_are_reported_per_provider():
    have = providers_with_keys({"ANTHROPIC_API_KEY": "k", "OPENROUTER_API_KEY": "", "GROQ_API_KEY": "k"})
    assert have["anthropic"] and have["groq"] and not have["openrouter"] and not have["openai"] and set(have) == set(PROVIDER_KEYS)


@pytest.mark.parametrize("env,want", [({}, "own_key"), ({"OPENROUTER_API_KEY": "k"}, "explore"), ({"OPENAI_API_KEY": "k"}, "own_key"),
                                      ({"OPENAI_API_KEY": "k", "OPENROUTER_API_KEY": "k"}, "own_key")])
def test_the_default_mode_follows_the_keys_you_have_and_prefers_your_own(env, want):
    assert default_mode(env) == want


def test_the_modes_describe_themselves_for_the_page_and_the_command_line():
    d = {m["key"]: m for m in describe_modes({"OPENROUTER_API_KEY": "k"})}
    assert d["explore"]["ready"] and not d["own_key"]["ready"] and d["explore"]["title"] == "Explore models" and d["own_key"]["title"] == "Use my own key"
    assert d["explore"]["providers"] == [{"name": "openrouter", "key_env": "OPENROUTER_API_KEY", "key_set": True}]
    assert "pass through OpenRouter" in d["explore"]["summary"] and "no middleman" in d["own_key"]["summary"]
    assert [p["name"] for p in d["own_key"]["providers"]][:2] == ["anthropic", "openai"] and all(not p["key_set"] for p in d["own_key"]["providers"])
