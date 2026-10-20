"""Document and code-project access are tools in the Tool Manager: they are registered, counted, listed in the prompt and permitted per agent
like any other tool. (They were briefly built-in context functions; this pins the arrangement that replaced that.)"""
import json

import pytest
from tests.fakes import FakeAPI
from tests.helpers import doc, make_service, make_zip, send

from pal.agents import AGENTS
from pal.chat.services import SERVICE_NAMES
from pal.chat.tools import ToolCall
from pal.sdk.agent import AgentSpec

DOC_TOOLS = {"search_documents", "read_document", "document_outline"}
PROJECT_TOOLS = {"read_source", "find_usages", "search_source", "list_project_files"}
KNOWLEDGE = {"file_to_markdown", "build_code_graph", "query_code_graph"}
CORE = {"calculator", "current_time", "count_tokens", "estimate_cost", "summarize_text", "save_file", "wikipedia", "arxiv_search"}


def run(tb, name, /, **args):
    return tb.run(ToolCall(name, args))


def call(name, /, **args):
    return f"<tool>{json.dumps({'name': name, 'args': args})}</tool>"


@pytest.fixture
def box(tmp_path):
    svc, _ = make_service(tmp_path)
    send(svc, "hello", route="api", sid="s")
    tb = svc.sessions.get("s").toolbox
    return svc, tb


def test_they_are_registered_with_the_tool_manager_in_every_session(box):
    _, tb = box
    registered = set(tb.manager.list_tools())
    assert registered == CORE | DOC_TOOLS | PROJECT_TOOLS | KNOWLEDGE | set(SERVICE_NAMES) and len(registered) == 28   # 8 core + 7 + 3 + 10


def test_they_are_offered_only_when_there_is_material_and_listed_in_the_one_tools_sentence(box):
    svc, tb = box
    assert not [n for n in tb.available("explain hashing") if n.split("(")[0] in DOC_TOOLS | PROJECT_TOOLS]
    svc.graph.ingest("s", "shop.zip", make_zip())
    send(svc, "see this", [doc("notes.md", "some notes about the shop")], route="api", sid="s")
    offered = {n.split("(")[0] for n in tb.available("explain checkout")}
    assert DOC_TOOLS | PROJECT_TOOLS <= offered
    p = tb.prompt("explain checkout")
    assert p.count("Tools:") == 2 and "find_usages(symbol)" in p and "document_outline(name)" in p and "Built in" not in p


def test_each_agent_may_use_exactly_the_tools_its_spec_lists_in_the_one_list():
    knowledge = {"query_code_graph", "build_code_graph", "file_to_markdown"}
    assert AGENTS["coding"].spec.tools == frozenset({"read_source", "search_documents", "read_document", "find_usages", "search_source", "list_project_files"} | knowledge)
    assert AGENTS["docs"].spec.tools == frozenset({"read_source", "search_documents", "read_document", "document_outline", "list_project_files", "search_source", "save_file"} | knowledge)
    assert AgentSpec(id="x", name="X", description="d", role="r").tools == frozenset({"read_source", "search_documents", "read_document"})
    assert not hasattr(AgentSpec(id="x", name="X", description="d", role="r"), "context")             # no second list


def test_the_allow_list_applies_to_them_like_any_tool(box):
    svc, tb = box
    svc.graph.ingest("s", "shop.zip", make_zip())
    send(svc, "see this", [doc("notes.md", "some notes about the shop")], route="api", sid="s")
    tb.allow = AGENTS["coding"].spec.tools
    assert run(tb, "find_usages", symbol="checkout").ok and not run(tb, "calculator", expression="1+1").ok
    assert not run(tb, "document_outline", name="notes.md").ok                                         # the coding agent has no outline tool
    tb.allow = AGENTS["docs"].spec.tools
    assert run(tb, "document_outline", name="notes.md").ok and not run(tb, "find_usages", symbol="checkout").ok
    tb.allow = None
    assert run(tb, "find_usages", symbol="checkout").ok and not hasattr(tb, "allow_context") and not hasattr(tb, "access")


def test_a_turn_reads_an_attached_document_with_a_tool_and_logs_it_as_a_tool_call(tmp_path):
    text = "\n\n".join(f"# Part {i}\n\n" + " ".join(f"w{i}x{j}" for j in range(500)) for i in range(1, 6))
    api = FakeAPI([call("read_document", name="handbook.md", part=2), "Part two is about w1x300."])
    svc, _ = make_service(tmp_path, api=api)
    out = send(svc, "what is in part 2 of the handbook?", [doc("handbook.md", text)], route="api")
    assert out.reply == "Part two is about w1x300." and [(t["name"], t["ok"]) for t in out.tools] == [("read_document", True)]
    assert "read_document(name, part)" in api.seen[0][0]["content"] and "Built in" not in api.seen[0][0]["content"]
    assert api.seen[1][-1]["content"].startswith('<tool_result name="read_document">')
