from pal.access_manager.pipeline import AccessManagerPipeline
from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMWrapper,
    WrapperCapabilities,
    WrapperConfig,
)
from pal.llm_wrapper.pipeline import LLMWrapperPipeline
from pal.sdk.pipeline import AgentSDK
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.tool_manager.components.base import ToolParam
from pal.tool_manager.pipeline import ToolManagerPipeline


class FakeCore(LLMWrapper):
    name = "fake"

    def __init__(self, config=None):
        self.config = config

    def chat(self, syscall):
        return LLMResponse(text=f"reply to: {syscall.messages[-1]['content']}")

    def token_count(self, text):
        return len(text.split())

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


def _build_sdk(tmp_path, agent_id="agent-1", syscall_scheduler=None, confirmation_handler=lambda r: True):
    am = AccessManagerPipeline(confirmation_handler=confirmation_handler)
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    tm = ToolManagerPipeline()
    llm_wrapper = LLMWrapperPipeline(WrapperConfig(backend="ollama", model="llama3"))
    llm_wrapper.core = FakeCore()
    return AgentSDK(agent_id, llm_wrapper, sm, tm, am, syscall_scheduler=syscall_scheduler), am, sm, tm


def test_chat_direct_without_scheduler(tmp_path):
    sdk, *_ = _build_sdk(tmp_path)
    reply = sdk.chat([{"role": "user", "content": "hello"}])
    assert reply == "reply to: hello"


def test_chat_through_shared_syscall_scheduler(tmp_path):
    from pal.orchestrator.components.syscall_scheduler import SyscallScheduler
    scheduler = SyscallScheduler()
    sdk, *_ = _build_sdk(tmp_path, syscall_scheduler=scheduler)

    reply = sdk.chat([{"role": "user", "content": "hi there"}])
    assert reply == "reply to: hi there"
    assert len(scheduler.requests) == 1


def test_storage_round_trip(tmp_path):
    sdk, *_ = _build_sdk(tmp_path)
    sdk.write("prefs.json", {"theme": "dark"})
    assert sdk.read("prefs.json") == {"theme": "dark"}
    assert "prefs.json" in sdk.list_keys()
    sdk.delete("prefs.json")
    assert sdk.read("prefs.json") is None


def test_memory_remember_and_recall(tmp_path):
    sdk, *_ = _build_sdk(tmp_path)
    sdk.remember("the user prefers tea over coffee")
    results = sdk.recall("tea")
    assert len(results) >= 1
    assert "tea" in results[0].content.lower()


def test_knowledge_base_search(tmp_path):
    sdk, *_ = _build_sdk(tmp_path)
    sdk.add_knowledge("doc1", "the weather in paris is sunny today")
    results = sdk.search_knowledge("weather paris")
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_call_tool(tmp_path):
    sdk, am, sm, tm = _build_sdk(tmp_path)
    tm.register_tool("double", lambda: (lambda x: x * 2), params=[ToolParam("x", int)])
    assert sdk.call_tool("double", x=21) == 42
    assert "double" in sdk.list_tools()


def test_call_tool_raises_on_failure(tmp_path):
    sdk, *_ = _build_sdk(tmp_path)
    sdk._tool_manager.register_tool("boom", lambda: (lambda: (_ for _ in ()).throw(ValueError("kaboom"))), params=[])
    try:
        sdk.call_tool("boom")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "kaboom" in str(e)


def test_cross_agent_storage_gated_by_access_manager(tmp_path):
    am = AccessManagerPipeline()  # default handler denies
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    tm = ToolManagerPipeline()
    llm_wrapper = LLMWrapperPipeline(WrapperConfig(backend="ollama", model="llama3"))
    llm_wrapper.core = FakeCore()

    sdk_a = AgentSDK("agent-a", llm_wrapper, sm, tm, am)
    sdk_b = AgentSDK("agent-b", llm_wrapper, sm, tm, am)

    sdk_a.write("secret.json", "a's data")
    try:
        sdk_b.read_from("agent-a", "secret.json")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "access denied" in str(e)

    am.assign_privilege_group("agent-a", "team-x")
    am.assign_privilege_group("agent-b", "team-x")
    assert sdk_b.read_from("agent-a", "secret.json") == "a's data"


def test_request_access_checks_without_performing(tmp_path):
    from pal.access_manager.components.base import OperationType
    am = AccessManagerPipeline()
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    tm = ToolManagerPipeline()
    llm_wrapper = LLMWrapperPipeline(WrapperConfig(backend="ollama", model="llama3"))
    llm_wrapper.core = FakeCore()
    sdk_a = AgentSDK("agent-a", llm_wrapper, sm, tm, am)

    decision = sdk_a.request_access("agent-b", OperationType.READ)
    assert not decision.allowed
    # nothing was actually attempted -- agent-b's storage untouched, no exception raised
