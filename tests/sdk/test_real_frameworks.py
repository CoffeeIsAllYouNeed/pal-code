"""Integration tests against the REAL third-party libraries. Each one skips itself when the library is not installed,
so the normal test run stays fast and offline; CI can install the libraries to run them."""
import warnings

import pytest
from tests.fakes import FakeAPI

from pal.sdk import AgentSDK, adapt


def sdk_for(tmp_path, replies):
    core = FakeAPI(replies)
    return AgentSDK.standalone(core, root=str(tmp_path / "sdk")), core


def test_real_autogen_02_two_agent_chat_runs_through_pal(tmp_path):
    pytest.importorskip("autogen.oai.client", reason="needs the AutoGen 0.2 line: pip install 'pyautogen==0.2.35'")
    import autogen
    sdk, core = sdk_for(tmp_path, ["Hello from the Pal-backed model."])
    cfg = {"config_list": [{"model": "gpt-x", "api_key": "not-used"}], "cache_seed": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assistant = autogen.AssistantAgent("assistant", llm_config=cfg, system_message="You are helpful.")
        user = autogen.UserProxyAgent("user", human_input_mode="NEVER", max_consecutive_auto_reply=0,
                                      code_execution_config=False)
        with adapt("autogen", sdk):
            res = user.initiate_chat(assistant, message="Say hello", max_turns=1, silent=True)
    assert "Hello from the Pal-backed model." in [m["content"] for m in res.chat_history]
    assert sdk.stats["calls"] == 1 and core.seen[0][0]["content"] == "You are helpful."


def test_real_open_interpreter_chat_runs_through_pal(tmp_path):
    pytest.importorskip("interpreter", reason="needs open-interpreter (0.4.x)")
    from interpreter import interpreter
    sdk, core = sdk_for(tmp_path, ["Hello from the Pal-backed model."])
    interpreter.auto_run, interpreter.offline = False, True
    interpreter.llm.model, interpreter.llm.api_key = "gpt-x", "unused"
    interpreter.llm.supports_functions = interpreter.llm.supports_vision = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with adapt("open-interpreter", sdk):
            out = interpreter.chat("Say hello", display=False, stream=False)
    assert "Hello from the Pal-backed model." in [m.get("content") for m in out if m.get("role") == "assistant"]
    assert sdk.stats["calls"] == 1 and core.seen[0][0]["role"] == "system"


def test_real_metagpt_exposes_the_method_the_adapter_patches(tmp_path):
    pytest.importorskip("metagpt.provider.base_llm", reason="needs metagpt")
    sdk, _ = sdk_for(tmp_path, ["x"])
    with adapt("metagpt", sdk) as ad:
        assert ad.installed
    assert not ad.installed
