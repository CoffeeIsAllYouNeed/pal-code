"""The routing rule: LLM queries go to the Pal kernel, non-LLM queries go to the OS kernel, and never crossed."""
import json

import pytest
from tests.fakes import FakeAPI

from pal.errors import UserInputError
from pal.os_kernel import OSKernelError
from pal.sdk import (
    OS_KERNEL,
    PAL_KERNEL,
    AgentSDK,
    LLMQuery,
    MemoryQuery,
    OSQuery,
    ReActAgent,
    StorageQuery,
    ToolQuery,
    register_os_tools,
)
from pal.sdk.queries import Query


@pytest.fixture
def rig(tmp_path):
    """An SDK whose model core and OS kernel are both spied on."""
    core = FakeAPI(["model says hi"])
    sdk = AgentSDK.standalone(core, root=str(tmp_path / "sdk"))
    os_calls = []
    real = sdk._os_kernel.handle
    sdk._os_kernel.handle = lambda agent, op, params=None: os_calls.append((agent, op)) or real(agent, op, params)
    return sdk, core, os_calls


def test_llm_queries_go_to_the_pal_kernel_and_never_touch_the_os(rig):
    sdk, core, os_calls = rig
    r = sdk.send_request(LLMQuery(messages=[{"role": "user", "content": "plan my day"}], max_tokens=50))
    assert r.ok and r.handled_by == PAL_KERNEL and r.data.text == "model says hi"
    assert len(core.seen) == 1 and os_calls == []


def test_os_queries_go_to_the_os_kernel_and_never_reach_a_model(rig, tmp_path):
    sdk, core, os_calls = rig
    w = sdk.send_request(OSQuery(op="file_write", params={"path": "notes/a.txt", "content": "hello"}))
    r = sdk.send_request(OSQuery(op="file_read", params={"path": "notes/a.txt"}))
    i = sdk.send_request(OSQuery(op="sys_info"))
    assert (w.ok, r.ok, i.ok) == (True, True, True) and r.data == "hello" and "kernel" in i.data
    assert {x.handled_by for x in (w, r, i)} == {OS_KERNEL}
    assert core.seen == [] and [c[1] for c in os_calls] == ["file_write", "file_read", "sys_info"]
    assert (tmp_path / "sdk" / "workspaces" / "agent" / "notes" / "a.txt").read_text(encoding="utf-8") == "hello"


def test_pal_kernel_managers_handle_memory_storage_and_tool_queries_not_the_os(rig):
    sdk, core, os_calls = rig
    replies = [
        sdk.send_request(MemoryQuery(op="remember", params={"content": "the sky is blue today"})),
        sdk.send_request(MemoryQuery(op="recall", params={"query": "sky colour"})),
        sdk.send_request(StorageQuery(op="write", params={"key": "k.json", "data": {"a": 1}})),
        sdk.send_request(StorageQuery(op="read", params={"key": "k.json"})),
        sdk.send_request(ToolQuery(tool="calculator", args={"expression": "6*7"})),
    ]
    assert all(r.ok and r.handled_by == PAL_KERNEL for r in replies)
    assert replies[3].data == {"a": 1} and replies[4].data == 42
    assert os_calls == [] and core.seen == []


def test_refusals_and_failures_come_back_as_responses_tagged_with_where_they_happened(rig):
    sdk, core, _ = rig
    refused = sdk.send_request(OSQuery(op="file_read", params={"path": "../../etc/passwd"}))
    assert not refused.ok and refused.handled_by == OS_KERNEL and "invalid path" in refused.error
    keyboard = sdk.send_request(OSQuery(op="keyboard"))
    assert not keyboard.ok and "desktop session" in keyboard.error
    core.boom = RuntimeError("model down")
    failed = sdk.send_request(LLMQuery(messages=[{"role": "user", "content": "x"}]))
    assert not failed.ok and failed.handled_by == PAL_KERNEL and "model down" in failed.error
    assert not sdk.send_request(StorageQuery(op="explode")).ok


def test_unknown_query_classes_are_rejected(rig):
    class Weird(Query):
        query_class = "weird"
    with pytest.raises(UserInputError, match="unknown query class"):
        rig[0].send_request(Weird())


def test_the_sdk_os_helper_returns_data_or_raises(rig):
    sdk, core, os_calls = rig
    sdk.os.write_file("a.txt", "x")
    assert sdk.os.read_file("a.txt") == "x" and sdk.os.list_files() == ["a.txt"] and sdk.os.system_info()["system"]
    with pytest.raises(OSKernelError, match="invalid path"):
        sdk.os.read_file("../nope")
    assert core.seen == [] and len(os_calls) == 5


def test_os_destructive_calls_ask_the_confirmation_gate(rig):
    sdk, _, _ = rig
    sdk.os.write_file("a.txt", "x")
    with pytest.raises(OSKernelError, match="not permitted"):
        sdk.os.delete_file("a.txt")                       # no confirmation handler: declined
    sdk._os_kernel.access.set_confirmation_handler(lambda req: True)
    assert sdk.os.delete_file("a.txt") == {"path": "a.txt", "deleted": True}


def test_os_tools_are_off_by_default_and_run_through_the_os_kernel_when_registered(rig):
    sdk, core, os_calls = rig
    assert not {"read_file", "write_file", "http_get"} & set(sdk.list_tools())
    assert register_os_tools(sdk) == ["read_file", "write_file", "list_files", "file_to_markdown", "http_get", "system_info"]
    core.replies = ['Thought: save it.\nAction: write_file {"path": "out.txt", "content": "done"}', "Final Answer: saved"]
    core.seen.clear()
    res = ReActAgent(sdk).run("save the word done to out.txt")
    assert res.answer == "saved" and sdk.os.read_file("out.txt") == "done"
    assert len(core.seen) == 2                            # the model only reasoned; the file write was an OS-kernel call
    assert ("agent", "file_write") in os_calls
    refused = ReActAgent(sdk)
    core.replies, core.seen[:] = ['Action: read_file {"path": "../../etc/passwd"}', "Final Answer: x"], []
    out = refused.run("read the password file")
    assert out.steps[0].observation.startswith("error") and "invalid path" in out.steps[0].observation


# -- the chat service wires one OS kernel interface into every session --------------------------------------------
def test_chat_sessions_share_the_os_kernel_and_chat_turns_never_use_it(tmp_path):
    from tests.helpers import make_service, send
    svc, api = make_service(tmp_path)
    send(svc, "explain the cache", route="api", sid="s1")
    assert list(svc.os_kernel.audit) == []                # a normal chat turn makes no OS call
    sdk = svc.sessions.get("s1").sdk
    sdk.os.write_file("n.txt", "x")
    assert (tmp_path / "home" / "workspaces" / "chat-s1" / "n.txt").is_file()
    assert [e["op"] for e in svc.os_kernel.audit] == ["file_write"]
    assert json.dumps(list(svc.os_kernel.audit)).count("\"x\"") == 0
