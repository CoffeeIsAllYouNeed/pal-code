"""The Local LLM component: models served on this machine, and the in-process Hugging Face model."""
from types import SimpleNamespace as NS

import pytest

from pal.llm_wrapper import DeploymentType, LLMSyscall, WrapperConfig, build_wrapper
from pal.llm_wrapper.components.litellm_wrapper import LiteLLMWrapper
from pal.local_llm import DEFAULT_URLS, LOCAL_BACKENDS, HuggingFaceLocalLLM, LocalLLM


def reply(payload=None, ok=True):
    class R:
        def raise_for_status(self):
            if not ok:
                raise OSError("503")

        def json(self):
            return payload
    return R()


def test_every_local_runtime_builds_a_local_llm_which_is_a_litellm_wrapper_underneath(fake_litellm):
    assert set(LOCAL_BACKENDS) == {"ollama", "vllm", "lmstudio", "llamacpp"} == set(DEFAULT_URLS)
    for name in LOCAL_BACKENDS:
        w = build_wrapper(WrapperConfig(backend=name, model="llama3.2:3b"))
        assert type(w) is LocalLLM and isinstance(w, LiteLLMWrapper) and w.local
        assert w.capabilities().deployment_type is DeploymentType.LOCAL_SERVER and w.capabilities().billed_per_token is False
    assert build_wrapper(WrapperConfig(backend="ollama", model="llama3.2:3b")).model == "ollama_chat/llama3.2:3b"


def test_a_cloud_backend_cannot_be_built_as_a_local_llm(fake_litellm):
    with pytest.raises(ValueError, match="'openai' is not a local runtime"):
        LocalLLM(WrapperConfig(backend="openai", model="gpt-4o", api_key="k"))


def test_a_local_model_is_free_has_a_long_first_call_timeout_and_never_borrows_a_real_key(fake_litellm, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-must-never-reach-a-local-server")
    w = build_wrapper(WrapperConfig(backend="vllm", model="x", host="http://localhost:8000/v1"))
    w.chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    kw = fake_litellm.calls[0]
    assert kw["api_key"] == "not-needed" and kw["timeout"] == 300.0 and w.price_per_token() is None
    assert "sk-real" not in str(kw)


def test_the_server_address_is_the_configured_one_or_the_runtimes_default(fake_litellm):
    assert build_wrapper(WrapperConfig(backend="ollama", model="m")).url == "http://localhost:11434"
    assert build_wrapper(WrapperConfig(backend="lmstudio", model="m")).url == "http://localhost:1234/v1"
    assert build_wrapper(WrapperConfig(backend="ollama", model="m", host="http://gpu-box:11434/")).url == "http://gpu-box:11434"
    assert build_wrapper(WrapperConfig(backend="vllm", model="m", base_url="http://x:9/v1")).url == "http://x:9/v1"


def test_ollama_reports_the_models_it_has_installed(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="ollama", model="m"))
    seen = []
    get = lambda url, timeout: seen.append((url, timeout)) or reply({"models": [{"name": "llama3.2:3b"}, {"model": "qwen2.5:7b"}, {}]})  # noqa: E731
    assert w.installed_models(get=get) == ["llama3.2:3b", "qwen2.5:7b"] and seen == [("http://localhost:11434/api/tags", 3.0)]


def test_the_openai_style_servers_report_their_models_from_the_models_endpoint(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="vllm", model="m", host="http://localhost:8000/v1"))
    seen = []
    get = lambda url, timeout: seen.append(url) or reply({"data": [{"id": "b-model"}, {"id": "a-model"}, {"nope": 1}]})  # noqa: E731
    assert w.installed_models(get=get) == ["a-model", "b-model"] and seen == ["http://localhost:8000/v1/models"]


def test_a_server_that_is_down_has_no_models_and_is_not_running_and_nothing_raises(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="ollama", model="m"))

    def down(url, timeout):
        raise ConnectionError("refused")
    assert w.installed_models(get=down) == [] and w.is_running(get=down) is False
    assert w.installed_models(get=lambda u, timeout: reply(ok=False)) == [] and w.is_running(get=lambda u, timeout: reply(ok=False)) is False
    assert w.installed_models(get=lambda u, timeout: reply("not json shape")) == []


def test_a_running_server_with_no_models_yet_is_still_running(fake_litellm):
    w = build_wrapper(WrapperConfig(backend="ollama", model="m"))
    get = lambda u, timeout: reply({"models": []})  # noqa: E731
    assert w.is_running(get=get) is True and w.installed_models(get=get) == []


def test_the_in_process_hugging_face_model_lives_in_this_component_too():
    from pal.llm_wrapper import LLMWrapper
    from pal.llm_wrapper.utils.registry import WRAPPER_REGISTRY
    assert WRAPPER_REGISTRY["huggingface_local"] is HuggingFaceLocalLLM and issubclass(HuggingFaceLocalLLM, LLMWrapper) and not issubclass(HuggingFaceLocalLLM, LiteLLMWrapper)
    assert HuggingFaceLocalLLM.__module__ == "pal.local_llm.components.huggingface_local" and NS
