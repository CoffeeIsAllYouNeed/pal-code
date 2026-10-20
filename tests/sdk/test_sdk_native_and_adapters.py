import asyncio
import sys
import time
import types

import pytest
from tests.fakes import FakeAPI

from pal.sdk import (
    AgentSDK,
    AgentSpec,
    CallSavers,
    FrameworkAdapterError,
    NativeAgent,
    ReActAgent,
    ReflexionAgent,
    adapt,
    supported_frameworks,
)
from pal.sdk.savers import tidy


@pytest.fixture
def sdk(tmp_path):
    def make(replies=None, **kw):
        return AgentSDK.standalone(FakeAPI(replies or ["ok"]), root=str(tmp_path / "sdk"), **kw)
    return make


# -- standalone SDK -----------------------------------------------------------------------------------------
def test_standalone_sdk_works_in_one_call_with_tools_storage_memory_and_stats(sdk):
    s = sdk(["hello"])
    assert s.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert s.call_tool("calculator", expression="6*7") == 42 and "calculator" in s.list_tools()
    s.write("k.json", {"a": 1})
    assert s.read("k.json") == {"a": 1}
    s.remember("the sky is blue today", tags=["t"])
    assert any("sky" in n.content for n in s.recall("sky colour"))
    assert s.stats["calls"] == 1 and s.stats["prompt_tokens"] == 10 and s.stats["completion_tokens"] == 3


# -- light savers ---------------------------------------------------------------------------------------------
def test_tidy_never_touches_code_blocks():
    text = "a   \n\n\n\n\nb\n```python\nx = 1   \n\n\n\ny = 2\n```\nc   \n\n\n\nd"
    out = tidy(text)
    assert out == "a\n\nb\n```python\nx = 1   \n\n\n\ny = 2\n```\nc\n\nd"


def test_savers_tidy_messages_and_count_the_characters(sdk):
    s = sdk()
    s.chat([{"role": "user", "content": "hi   \n\n\n\nthere"}])
    assert s.stats["chars_tidied"] == 5 and s.savers.stats.calls == 1


def test_repeat_cache_is_off_by_default_and_exact_when_on(sdk):
    s = sdk(["one", "two", "three"])
    m = [{"role": "user", "content": "same"}]
    assert [s.chat(m), s.chat(m)] == ["one", "two"] and s.stats["cache_hits"] == 0
    c = sdk(["one", "two", "three"], savers=CallSavers(cache=True))
    assert c.chat(m) == "one" and c.chat(m) == "one" and c.stats["cache_hits"] == 1
    assert c.chat(m, max_tokens=5) == "two"                       # different request, different key
    r = c.chat_response(m)
    assert r.prompt_tokens == 0 and r.completion_tokens == 0 and r.meta["cache"] is True


def test_cache_entries_expire(sdk):
    c = sdk(["one", "two"], savers=CallSavers(cache=True, cache_ttl=0.05))
    m = [{"role": "user", "content": "same"}]
    assert c.chat(m) == "one"
    time.sleep(0.08)
    assert c.chat(m) == "two"


# -- native agents -------------------------------------------------------------------------------------------------
class Echo(NativeAgent):
    spec = AgentSpec(id="echo", name="Echo", description="d", role="Role: echo.", max_tokens=77)


def test_native_agent_without_a_kernel_makes_one_sdk_call_with_its_role(sdk):
    s = sdk(["pong"])
    res = Echo(s).run("ping")
    assert res.answer == "pong" and res.ok
    seen = s._llm_wrapper.core.seen[-1]
    assert seen[0] == {"role": "system", "content": "Role: echo."} and seen[1]["content"] == "ping"


def test_native_agent_with_a_kernel_uses_the_full_pipeline():
    calls = []

    class Kernel:
        def send(self, sid, provider, model, key, message, attachments, **kw):
            calls.append((message, kw))
            return types.SimpleNamespace(reply="done", input_tokens=5, output_tokens=2)

    res = Echo(kernel=Kernel()).run("ping")
    assert res.answer == "done" and res.meta == {"input_tokens": 5, "output_tokens": 2} and calls == [("ping", {"agent": "echo"})]


def test_native_agent_needs_a_spec_or_its_own_run(sdk):
    with pytest.raises(NotImplementedError):
        NativeAgent(sdk()).run("x")
    with pytest.raises(RuntimeError, match="no SDK or kernel"):
        Echo().run("x")


# -- ReAct ---------------------------------------------------------------------------------------------------------
def react(sdk, replies, **kw):
    s = sdk(replies)
    a = ReActAgent(s)
    for k, v in kw.items():
        setattr(a, k, v)
    return a, s


def test_react_reasons_acts_observes_and_answers(sdk):
    a, s = react(sdk, ['Thought: I need to multiply.\nAction: calculator {"expression": "6*7"}', "Thought: done\nFinal Answer: 42"])
    res = a.run("what is 6 times 7?")
    assert res.ok and res.answer == "42" and len(res.steps) == 1
    step = res.steps[0]
    assert (step.tool, step.args, step.observation) == ("calculator", {"expression": "6*7"}, "42")
    assert "calculator" in s._llm_wrapper.core.seen[0][0]["content"]
    assert s._llm_wrapper.core.seen[1][-1] == {"role": "user", "content": "Observation: 42"}


def test_react_ignores_invented_observations_and_handles_bad_calls(sdk):
    a, s = react(sdk, ['Thought: try\nAction: nope {}\nObservation: made up', "Action: calculator {bad json}",
                       'Action: calculator {"expression": "1+1"}', "Final Answer: 2"])
    res = a.run("task")
    assert res.answer == "2" and [x.observation.startswith("error") for x in res.steps] == [True, True, False]


def test_react_without_markers_returns_the_reply_and_stops_at_the_step_limit(sdk):
    a, _ = react(sdk, ["just an answer"])
    assert a.run("q").answer == "just an answer"
    b, _ = react(sdk, ['Action: calculator {"expression": "1+1"}'], max_steps=2)
    res = b.run("q")
    assert not res.ok and "no final answer after 2 steps" in res.error and len(res.steps) == 2


def test_react_can_be_limited_to_named_tools(sdk):
    a, s = react(sdk, ["Final Answer: x"], tools=["calculator"])
    a.run("q")
    system = s._llm_wrapper.core.seen[0][0]["content"]
    assert "calculator" in system and "word_count" not in system


# -- Reflexion -----------------------------------------------------------------------------------------------------
def test_reflexion_retries_with_a_lesson_and_remembers_it(sdk):
    s = sdk(["wrong answer", "Go through each step in order.", "right answer"])
    a = ReflexionAgent(s, evaluator=lambda task, ans: (ans == "right answer", "off by one"))
    res = a.run("solve it")
    assert res.ok and res.answer == "right answer" and [t.ok for t in res.steps] == [False, True]
    second_prompt = s._llm_wrapper.core.seen[2][0]["content"]
    assert "Lessons from earlier attempts" in second_prompt and "Go through each step in order." in second_prompt
    assert any("each step" in n.content for n in s.recall("step order"))


def test_reflexion_gives_up_after_max_trials_and_returns_the_last_answer(sdk):
    s = sdk(["a1", "lesson", "a2", "lesson", "a3", "lesson"])
    a = ReflexionAgent(s, evaluator=lambda t, ans: False)
    a.max_trials = 3
    res = a.run("x")
    assert not res.ok and res.answer == "a3" and len(res.steps) == 3 and "not solved in 3 trials" in res.error


def test_reflexion_can_judge_itself_and_use_any_native_agent_as_actor(sdk):
    s = sdk(["short answer", "FAIL: missing detail", "Add the detail.", "full answer", "PASS"])
    res = ReflexionAgent(s).run("explain")
    assert res.ok and res.answer == "full answer" and res.steps[0].feedback.startswith("FAIL")
    s2 = sdk(["Final Answer: 4", "PASS"])   # ReAct actor answers, then the judge passes it
    res2 = ReflexionAgent(s2, actor=ReActAgent(s2)).run("2+2")
    assert res2.ok and res2.answer == "4"


# -- non-native frameworks: fake libraries so the adapters really run ------------------------------------------------
@pytest.fixture
def fake_autogen(monkeypatch):
    class OpenAIWrapper:
        def create(self, **kw):
            return "REAL"
    mods = {"autogen": types.ModuleType("autogen"), "autogen.oai": types.ModuleType("autogen.oai"),
            "autogen.oai.client": types.ModuleType("autogen.oai.client")}
    mods["autogen.oai.client"].OpenAIWrapper = OpenAIWrapper
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return OpenAIWrapper


@pytest.fixture
def fake_interpreter(monkeypatch):
    mod = types.ModuleType("interpreter")
    mod.interpreter = types.SimpleNamespace(llm=types.SimpleNamespace(completions=lambda **kw: iter(["REAL"])))
    monkeypatch.setitem(sys.modules, "interpreter", mod)
    return mod.interpreter.llm


@pytest.fixture
def fake_metagpt(monkeypatch):
    class BaseLLM:
        async def acompletion_text(self, messages, stream=False, timeout=None):
            return "REAL"
    names = ["metagpt", "metagpt.provider", "metagpt.provider.base_llm"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["metagpt.provider.base_llm"].BaseLLM = BaseLLM
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return BaseLLM


def test_autogen_calls_go_through_pal_and_are_restored(sdk, fake_autogen):
    s = sdk(["from pal"])
    original = fake_autogen.create
    with adapt("autogen", s) as ad:
        assert ad.installed
        reply = fake_autogen().create(messages=[{"role": "user", "content": "hi"}], model="gpt-x")
        assert reply.choices[0].message.content == "from pal" and reply.model == "gpt-x"
        # what real AutoGen 0.2 reads back from every response
        assert reply.message_retrieval_function(reply) == ["from pal"] and reply.cost == 0.0
        assert (reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens) == (10, 3, 13)
    assert fake_autogen.create is original and fake_autogen().create() == "REAL"
    assert s.stats["calls"] == 1


def test_open_interpreter_calls_go_through_pal_and_are_restored(sdk, fake_interpreter):
    s = sdk(["from pal"])
    with adapt("open-interpreter", s):
        chunks = list(fake_interpreter.completions(messages=[{"role": "user", "content": "run it"}]))
        assert chunks == [{"choices": [{"delta": {"content": "from pal"}}]}]
    assert list(fake_interpreter.completions()) == ["REAL"]


def test_metagpt_calls_go_through_pal_without_blocking_the_event_loop(tmp_path, fake_metagpt):
    class Slow(FakeAPI):
        def chat(self, syscall):
            time.sleep(0.3)
            return super().chat(syscall)

    s = AgentSDK.standalone(Slow(["from pal"]), root=str(tmp_path / "s"))
    ticks = []

    async def ticker():
        for _ in range(5):
            await asyncio.sleep(0.05)
            ticks.append(time.monotonic())

    async def main():
        with adapt("metagpt", s):
            call = fake_metagpt().acompletion_text([{"role": "user", "content": "plan"}])
            return await asyncio.gather(call, ticker())

    reply, _ = asyncio.run(main())
    assert reply == "from pal" and len(ticks) == 5            # the loop kept running during the 0.3 s model call
    assert asyncio.run(fake_metagpt().acompletion_text([])) == "REAL"


def test_two_sdks_cannot_adapt_the_same_framework_at_once(sdk, fake_autogen):
    a, b = adapt("autogen", sdk(["x"])), adapt("autogen", sdk(["y"]))
    a.install()
    with pytest.raises(FrameworkAdapterError, match="already adapted"):
        b.install()
    a.uninstall()
    b.install()
    b.uninstall()
    assert fake_autogen().create() == "REAL"


def test_a_changed_framework_api_fails_loudly(sdk, monkeypatch):
    mod = types.ModuleType("autogen.oai.client")
    mod.OpenAIWrapper = type("OpenAIWrapper", (), {})
    for k in ("autogen", "autogen.oai"):
        monkeypatch.setitem(sys.modules, k, types.ModuleType(k))
    monkeypatch.setitem(sys.modules, "autogen.oai.client", mod)
    with pytest.raises(FrameworkAdapterError, match="no 'create' method"):
        adapt("autogen", sdk()).install()


# -- registry --------------------------------------------------------------------------------------------------------
def test_registry_names_aliases_and_helpful_errors(sdk):
    assert supported_frameworks() == ["autogen", "open-interpreter", "metagpt", "react", "reflexion"]
    assert type(adapt("Open_Interpreter", sdk())).__name__ == "OpenInterpreterAdapter"
    assert type(adapt("pyautogen", sdk())).__name__ == "AutoGenAdapter"
    with pytest.raises(FrameworkAdapterError, match="use ReActAgent"):
        adapt("react", sdk())
    with pytest.raises(FrameworkAdapterError, match="use ReflexionAgent"):
        adapt("Reflexion", sdk())
    with pytest.raises(FrameworkAdapterError, match="supported: autogen"):
        adapt("langchain", sdk())


# -- AutoGen 0.4+ (model client, no monkeypatching) -------------------------------------------------------------------
def test_new_autogen_model_client_runs_an_agent_through_pal(sdk):
    pytest.importorskip("autogen_agentchat")
    from autogen_agentchat.agents import AssistantAgent

    from pal.sdk import autogen_model_client
    s = sdk(["hello from pal"])
    agent = AssistantAgent("helper", model_client=autogen_model_client(s), system_message="You are helpful.")
    res = asyncio.run(agent.run(task="say hello"))
    assert str(res.messages[-1].content) == "hello from pal" and s.stats["calls"] == 1
    assert s._llm_wrapper.core.seen[0][0] == {"role": "system", "content": "You are helpful."}


def test_new_autogen_client_gives_a_clear_error_when_the_library_is_missing(sdk, monkeypatch):
    from pal.sdk import autogen_model_client
    monkeypatch.setitem(sys.modules, "autogen_core", None)
    monkeypatch.setitem(sys.modules, "autogen_core.models", None)
    with pytest.raises(FrameworkAdapterError, match="autogen-agentchat"):
        autogen_model_client(sdk())
