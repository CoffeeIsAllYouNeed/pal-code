"""Any AI, any model: LiteLLM's whole catalogue, any OpenAI-compatible endpoint, local servers, and a model per agent."""
import pytest

from pal.terminal import backend
from pal.terminal.clients import PalClient
from pal.terminal.session import PalSession
from tests.pal.fakes import ScriptedCore


@pytest.mark.parametrize("spec,expect", [
    ("qwen3:8b", ("local", "qwen3:8b")), ("local:qwen3:8b", ("local", "qwen3:8b")), ("openai:gpt-5", ("openai", "gpt-5")),
    ("litellm:xai/grok-4", ("litellm", "xai/grok-4")), ("custom:my-model", ("custom", "my-model")), ("lmstudio:qwen", ("lmstudio", "qwen")),
])
def test_a_model_is_named_provider_colon_model(spec, expect):
    assert backend.split_spec(spec) == expect


def test_litellm_reaches_any_model_it_knows(fake_litellm):
    core = backend.make_core("litellm", "xai/grok-4")
    assert core.model == "xai/grok-4"


def test_a_custom_endpoint_needs_an_address_and_never_gets_an_openai_key(fake_litellm, monkeypatch, tmp_path):
    from pal.errors import UserInputError
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.delenv("PAL_CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("PAL_CUSTOM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    with pytest.raises(UserInputError, match="/endpoint"):
        backend.make_core("custom", "m")
    backend.save_prefs(custom_base_url="https://gateway.example/v1")
    core = backend.make_core("custom", "m")
    kw = core._kwargs(__import__("pal.llm_wrapper.components.base", fromlist=["LLMSyscall"]).LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    assert kw["api_base"] == "https://gateway.example/v1" and kw["api_key"] == "not-needed"
    monkeypatch.setenv("PAL_CUSTOM_API_KEY", "gw-key")
    core = backend.make_core("custom", "m")
    assert core._kwargs(__import__("pal.llm_wrapper.components.base", fromlist=["LLMSyscall"]).LLMSyscall(messages=[]))["api_key"] == "gw-key"


def test_local_servers_get_their_default_address(fake_litellm):
    core = backend.make_core("lmstudio", "qwen")
    assert "1234" in (core.config.host or "")


def test_the_endpoint_address_is_saved_but_keys_never_are(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    backend.save_prefs(custom_base_url="https://x/v1", model_plan="anthropic:claude", api_key="SECRET", PAL_CUSTOM_API_KEY="SECRET")
    saved = backend.prefs_path().read_text()
    assert "https://x/v1" in saved and "anthropic:claude" in saved and "SECRET" not in saved
    backend.save_prefs(model_plan="")
    assert "model_plan" not in backend.load_prefs()


class Named(ScriptedCore):
    def __init__(self, name, seen):
        super().__init__()
        self.name_, self.seen = name, seen

    def chat(self, syscall):
        from pal.llm_wrapper.components.base import LLMResponse
        self.seen.append(self.name_)
        return LLMResponse(f"from {self.name_}", prompt_tokens=10, completion_tokens=2)


def test_each_agent_can_use_its_own_model_in_one_conversation(tmp_path):
    seen = []
    client = PalClient(Named("main", seen), None, frozenset(), home=tmp_path / "e", local=False)
    s = PalSession(client, "main", "local", tmp_path, "ask")
    from pal.terminal.cli import set_role_model
    assert set_role_model(s, "plan", "openai:big", core_factory=lambda p, m, h: Named("planner", seen)) == ""
    try:
        s.ask("hello")
        s.mode = "plan"
        t = s.ask("plan it")
        s.mode = "ask"
        s.ask("and now?")
    finally:
        client.close()
    assert seen == ["main", "planner", "main"] and "from planner" in t.text and s.model_for("plan") == ("openai", "big")
    assert t.prompt_tokens == 10 and s.model_for("build") == ("local", "main")
    set_role_model(s, "plan", "default")
    assert "plan" not in s.role_models


def test_model_command_for_an_agent_is_saved_and_applied_next_time(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pal.terminal.cli import apply_role_models, handle
    from pal.terminal.ui import make_console
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    seen = []
    factory = lambda p, m, h: Named(f"{p}:{m}", seen)  # noqa: E731
    client = PalClient(Named("main", seen), None, frozenset(), home=tmp_path / "e", local=False)
    s = PalSession(client, "main", "local", tmp_path, "build")
    args = SimpleNamespace(host=None)
    handle("/model build litellm:xai/grok-4", args, s, make_console(), factory)
    assert backend.load_prefs()["model_build"] == "litellm:xai/grok-4" and s.model_for("build") == ("litellm", "xai/grok-4")
    client.close()
    client2 = PalClient(Named("main", seen), None, frozenset(), home=tmp_path / "e2", local=False)
    s2 = PalSession(client2, "main", "local", tmp_path, "build")
    apply_role_models(s2, factory)
    assert s2.model_for("build") == ("litellm", "xai/grok-4")
    client2.close()
