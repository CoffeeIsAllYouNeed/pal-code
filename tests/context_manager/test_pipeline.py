import tempfile

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMWrapper,
    WrapperCapabilities,
    WrapperConfig,
)
from pal.pipeline import Pipeline


class FakeCore(LLMWrapper):
    name = "fake"

    def chat(self, syscall):
        return LLMResponse(text="hi there", prompt_tokens=5, completion_tokens=2)

    def token_count(self, text):
        return len(text.split())

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


def test_pipeline_run():
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = Pipeline(WrapperConfig(backend="ollama", model="llama3"), memory_store_path=f"{tmp}/memory.json")
        pipeline.llm_wrapper.core = FakeCore(pipeline.llm_wrapper.config)

        out = pipeline.run("system prompt", "hello world")
        assert out == "hi there"
        assert pipeline.metrics.summary()["requests"] == 1
        assert len(pipeline.history) == 2
