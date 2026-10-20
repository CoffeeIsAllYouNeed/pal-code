import pytest

from pal.llm_wrapper import DeploymentType
from pal.llm_wrapper.components.litellm_wrapper import LiteLLMWrapper
from pal.llm_wrapper.components.openrouter_wrapper import OpenRouterWrapper
from pal.llm_wrapper.pipeline import LLMWrapperPipeline
from pal.llm_wrapper.utils.registry import WRAPPER_REGISTRY, WrapperConfig, available_backends, build_wrapper
from pal.local_llm.components.huggingface_local import HuggingFaceLocalLLM
from pal.local_llm.components.local_llm import LocalLLM


def test_registry_builds_a_litellm_core_for_ollama(fake_litellm):
    core = build_wrapper(WrapperConfig(backend="ollama", model="llama3"))
    assert isinstance(core, LiteLLMWrapper) and core.model == "ollama_chat/llama3" and core.local


def test_registry_builds_local_servers_with_no_key_and_marks_them_free(fake_litellm):
    core = build_wrapper(WrapperConfig(backend="llamacpp", model="local", host="http://localhost:8080/v1"))
    assert isinstance(core, LiteLLMWrapper) and core.model == "openai/local"
    caps = core.capabilities()
    assert caps.billed_per_token is False and caps.deployment_type is DeploymentType.LOCAL_SERVER


def test_pipeline_switch_backend(fake_litellm):
    p = LLMWrapperPipeline(WrapperConfig(backend="ollama", model="llama3"))
    assert p.core.model == "ollama_chat/llama3"
    p.switch(WrapperConfig(backend="llamacpp", model="local", host="http://localhost:8080/v1"))
    assert p.core.model == "openai/local"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        build_wrapper(WrapperConfig(backend="not-a-real-backend"))


def test_every_backend_but_in_process_hugging_face_goes_through_litellm():
    assert WRAPPER_REGISTRY["huggingface_local"] is HuggingFaceLocalLLM
    assert {n for n, c in WRAPPER_REGISTRY.items() if not issubclass(c, LiteLLMWrapper)} == {"huggingface_local"}        # all the rest are LiteLLM underneath
    assert {n for n, c in WRAPPER_REGISTRY.items() if c is LiteLLMWrapper} == {"openai", "azure_openai", "anthropic", "gemini", "groq", "together", "mistral",
                                                                             "deepseek", "perplexity", "bedrock", "litellm"}               # your own key for a provider
    assert {n for n, c in WRAPPER_REGISTRY.items() if c is LocalLLM} == {"ollama", "vllm", "lmstudio", "llamacpp"}                    # the Local LLM component
    assert {n for n, c in WRAPPER_REGISTRY.items() if c is OpenRouterWrapper} == {"openrouter"}                                      # one key, many models
    assert {"ollama", "vllm", "lmstudio", "llamacpp", "anthropic", "gemini", "openrouter", "litellm"} <= set(available_backends())
