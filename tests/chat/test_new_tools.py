"""The tools added for the coding and documentation agents, with the guards that matter most: what a model can make a tool do is
decided by the code, not by the model."""
import json
import re

import pytest
from tests.fakes import FakeAPI
from tests.helpers import client, doc, make_service, make_zip, send

from pal.agents import AGENTS
from pal.config import ToolsConfig
from pal.os_kernel import OSKernelError

BAD = r"^error"


class FakeOS:
    """Stands in for the OS kernel interface: records what it was asked and answers from a script."""

    def __init__(self, replies=None, deny=None):
        self.calls, self.replies, self.deny = [], replies or {}, deny

    def workspace(self, agent_id):
        files = getattr(self, "files", [])
        return type("WS", (), {"list": staticmethod(lambda prefix="": [f for f in files if f.startswith(prefix)])})()

    def handle(self, agent_id, op, params=None):
        self.calls.append((agent_id, op, dict(params or {})))
        if self.deny and op == "file_write" and self.deny in str(params.get("path")):
            raise OSKernelError(f"overwrite of {params['path']!r} was not permitted: denied")
        if op == "file_write":
            return {"path": params["path"], "bytes": len(params["content"])}
        return self.replies.get(op, {"status": 200, "text": "", "content_type": "", "truncated": False})


def box(os=None, allow=None, **cfg):
    from pal.chat.tools import ToolBox
    b = ToolBox("chat-s", ToolsConfig(**cfg), lambda t: len(t.split()), session_id="s", os_kernel=os)
    b.allow = allow
    return b


def nets(fos):
    """The network calls only: showing a tool to the model may also list the workspace, which is not a request."""
    return [c for c in fos.calls if c[1] == "net_get"]


def run(b, name, /, **args):
    from pal.chat.tools import ToolCall
    return b.run(ToolCall(name, args))


def own(b, name, /, **args):
    """The result text of a document or project tool, called through the tool box like any other."""
    return run(b, name, **args).text


# -- save_file: only for agents that list it ---------------------------------------------------------------------------------
def test_plain_chat_can_never_write_a_file_whatever_the_model_asks():
    fos = FakeOS()
    b = box(fos, allow=None)
    out = run(b, "save_file", path="README.md", content="hello")
    assert not out.ok and re.match(BAD, out.text) and "not allowed to save files" in out.text and fos.calls == [] and b.saved_files == []
    assert "save_file(path, content)" not in " ".join(b.available("write a readme"))        # not even offered


def test_an_agent_that_lists_save_file_writes_a_new_file_through_the_os_kernel_interface():
    fos = FakeOS()
    b = box(fos, allow=AGENTS["docs"].spec.tools)
    assert "save_file(path, content)" in b.available("write a readme")
    out = run(b, "save_file", path="docs/README.md", content="# Title")
    assert out.ok and "saved docs/README.md" in out.text and "download" in out.text
    assert fos.calls == [("chat-s", "file_write", {"path": "docs/README.md", "content": "# Title"})] and b.saved_files == ["docs/README.md"]


def test_save_file_is_refused_for_an_agent_without_it_and_when_switched_off():
    for b in (box(FakeOS(), allow=AGENTS["coding"].spec.tools),                              # the agent does not list it
              box(FakeOS(), allow=AGENTS["docs"].spec.tools, save_files=False)):             # the setting is off
        out = run(b, "save_file", path="a", content="b")
        assert re.match(BAD, out.text) and not out.ok and b.os_kernel.calls == [] and b.saved_files == []


def test_an_overwrite_that_is_declined_tells_the_model_to_pick_another_name_and_other_failures_are_clear():
    b = box(FakeOS(deny="README.md"), allow=AGENTS["docs"].spec.tools)
    out = run(b, "save_file", path="README.md", content="x")
    assert not out.ok and "not permitted" in out.text and "Choose a different file name" in out.text and b.saved_files == []
    missing = run(box(None, allow=AGENTS["docs"].spec.tools), "save_file", path="a", content="b")
    assert not missing.ok and re.match(BAD, missing.text) and "OS kernel interface" in missing.text


# -- the network lookups: off by default, two fixed sites -------------------------------------------------------------------------
def test_wikipedia_and_arxiv_are_off_by_default_and_never_reach_the_network():
    fos = FakeOS()
    b = box(fos)
    for name, args in (("wikipedia", {"title": "Pune"}), ("arxiv_search", {"query": "graph neural networks"})):
        out = run(b, name, **args)
        assert "turned off" in out.text and not out.ok or "turned off" in out.text
    assert fos.calls == [] and not [n for n in b.available("who was Ada Lovelace? any papers on her research?") if "wiki" in n or "arxiv" in n]


def test_wikipedia_uses_only_the_wikipedia_host_and_handles_missing_articles():
    fos = FakeOS({"net_get": {"status": 200, "text": json.dumps({"title": "Pune", "description": "city in India", "extract": "Pune is a city."})}})
    b = box(fos, network=True)
    assert "who was" and "wikipedia(title)" in b.available("who is Ada Lovelace")
    out = run(b, "wikipedia", title="Pune city")
    assert out.text.startswith("Pune: city in India") and "Pune is a city." in out.text
    url = nets(fos)[0][2]["url"]
    assert url == "https://en.wikipedia.org/api/rest_v1/page/summary/Pune_city" and nets(fos)[0][1] == "net_get"
    fos.replies["net_get"] = {"status": 404, "text": ""}
    assert "no Wikipedia article" in run(b, "wikipedia", title="Zzzz").text
    fos.replies["net_get"] = {"status": 200, "text": "<html>not json"}
    assert "unreadable" in run(b, "wikipedia", title="Pune").text


@pytest.mark.parametrize("title", ["a\x00b", "x" * 200, "evil<script>", "back\\slash", ""])
def test_a_wikipedia_title_cannot_smuggle_a_url_or_control_characters(title):
    fos = FakeOS()
    assert "plain article title" in run(box(fos, network=True), "wikipedia", title=title).text and fos.calls == []


def test_an_injected_title_is_percent_encoded_not_interpreted():
    fos = FakeOS({"net_get": {"status": 200, "text": "{}"}})
    run(box(fos, network=True), "wikipedia", title="a/../b?c=d#e")
    url = nets(fos)[0][2]["url"]
    assert url.startswith("https://en.wikipedia.org/api/rest_v1/page/summary/") and "?" not in url and "#" not in url and "../" not in url


ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.0001v1</id><published>2024-01-02T00:00:00Z</published><title>  Graph   Nets </title>
<summary>A study of graphs.</summary><author><name>A One</name></author><author><name>B Two</name></author></entry></feed>"""


def test_arxiv_results_are_parsed_and_the_query_is_encoded_and_clamped():
    fos = FakeOS({"net_get": {"status": 200, "text": ATOM}})
    b = box(fos, network=True)
    assert "arxiv_search(query, max_results)" in b.available("find papers on graph nets")
    out = run(b, "arxiv_search", query="graph & nets", max_results=99)
    assert "Graph Nets (2024) by A One, B Two" in out.text and "A study of graphs." in out.text and "2401.0001" in out.text
    url = nets(fos)[0][2]["url"]
    assert url.startswith("https://export.arxiv.org/api/query?search_query=all:graph%20%26%20nets&max_results=5")
    run(b, "arxiv_search", query="x", max_results=0)
    assert "max_results=1&" in nets(fos)[1][2]["url"]
    fos.replies["net_get"] = {"status": 200, "text": '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'}
    assert "no papers found" in run(b, "arxiv_search", query="zzzz").text


def test_arxiv_refuses_documents_that_define_entities_and_bad_responses():
    bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><feed xmlns="http://www.w3.org/2005/Atom">&b;</feed>'
    fos = FakeOS({"net_get": {"status": 200, "text": bomb}})
    b = box(fos, network=True)
    assert "unexpected" in run(b, "arxiv_search", query="x").text
    fos.replies["net_get"] = {"status": 200, "text": "<feed><entry>"}
    assert "unreadable" in run(b, "arxiv_search", query="x").text
    fos.replies["net_get"] = {"status": 503, "text": ""}
    assert "status 503" in run(b, "arxiv_search", query="x").text
    assert "give a search query" in run(b, "arxiv_search", query="   ").text


def test_a_blocked_network_request_comes_back_as_an_error_the_model_can_read():
    class Blocked(FakeOS):
        def handle(self, *a, **k):
            raise OSKernelError("host 'en.wikipedia.org' is not on the allowed list")
    out = run(box(Blocked(), network=True), "wikipedia", title="Pune")
    assert out.ok is False and "allowed list" in out.text


# -- utilities ---------------------------------------------------------------------------------------------------------------------
def test_token_cost_and_summary_tools_and_when_they_are_offered():
    b = box()
    assert "3 tokens" in run(b, "count_tokens", text="one two three").text
    assert "no price is known" in run(b, "estimate_cost", input_tokens=1000).text
    b.price_fn = lambda: (3e-6, 15e-6)
    out = run(b, "estimate_cost", input_tokens=1_000_000, output_tokens=100_000).text
    assert "$4.500000" in out and "$3.00 per million input" in out and "$15.00 per million output" in out
    assert "no local model" in run(b, "summarize_text", text="long text").text
    b.summarizer = lambda t: f"short({len(t)})"
    assert run(b, "summarize_text", text="x" * 9000).text == "short(6000)"                 # the input is capped
    assert "count_tokens(text)" in b.available("how many tokens is this?")
    assert "estimate_cost(input_tokens, output_tokens)" in b.available("how many tokens and what will it cost?")
    assert "estimate_cost(input_tokens, output_tokens)" not in b.available("how many tokens is this?")
    assert "summarize_text(text)" in b.available("please summarise this") and "summarize_text(text)" not in box().available("please summarise this")


# -- code navigation and the document outline, in a real session -------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    svc, api = make_service(tmp_path)
    svc.graph.ingest("session-1", "shop.zip", make_zip())
    send(svc, "hello there", route="api")                      # makes the session so it owns a toolbox bound to the graph
    return svc, svc.sessions.get("session-1").toolbox


def test_find_usages_separates_definitions_from_uses_and_rejects_odd_names(project):
    _, tb = project
    out = own(tb, "find_usages", symbol="checkout")
    assert out.startswith("checkout: 1 definition(s), 0 other use(s), in 1 file(s)") and "definition: shop/pay.py:4: def checkout(" in out
    uses = own(tb, "find_usages", symbol="charge_card")
    assert "1 definition(s), 1 other use(s)" in uses and "shop/pay.py:6:" in uses
    assert "no match" in own(tb, "find_usages", symbol="nonexistent_symbol")
    for bad in ("a b", "x; rm -rf /", "../etc", "(?i)checkout"):
        assert "plain name" in own(tb, "find_usages", symbol=bad)


def test_search_and_listing_stay_inside_the_project(project):
    _, tb = project
    assert "shop/cart.py:5:" in own(tb, "search_source", text="def add") and "no results" in own(tb, "search_source", text="zzzzzz")
    assert "at least two characters" in own(tb, "search_source", text="a")
    listing = own(tb, "list_project_files")
    assert "shop/cart.py (" in listing and "shop/pay.py (" in listing and "shop/unrelated.py (" in listing
    assert "shop/cart.py" not in own(tb, "list_project_files", prefix="shop/pay") and "no files match" in own(tb, "list_project_files", prefix="nope/")
    nothing = run(box(), "find_usages", symbol="x")                                   # nothing bound: the tool does not even exist
    assert not nothing.ok and "unknown tool" in nothing.text


def test_document_outline_lists_parts_with_sizes_and_is_case_insensitive(tmp_path):
    svc, _ = make_service(tmp_path)
    text = "\n\n".join(f"# Part {i}\n\n" + " ".join(f"w{i}x{j}" for j in range(400)) for i in range(1, 4))
    send(svc, "read this", [doc("Handbook.md", text)], route="api")
    tb = svc.sessions.get("session-1").toolbox
    out = own(tb, "document_outline", name="handbook.md")
    assert out.startswith("Handbook.md:") and "part 1 (" in out and ": Part 1" in out
    assert "no document named" in own(tb, "document_outline", name="missing.md") and "Handbook.md" in own(tb, "document_outline", name="missing.md")


# -- through a whole chat turn and the web layer ------------------------------------------------------------------------------------
def test_the_documentation_agent_saves_a_file_in_a_turn_and_the_user_can_download_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_DOCS", "k")
    call = '<tool>{"name": "save_file", "args": {"path": "README.md", "content": "# Papertrail\\nSearches notes."}}</tool>'
    api = FakeAPI([call, "I saved the README for you."])
    svc, _ = make_service(tmp_path, api=api, scheduler={"api_min_interval": {"gemini": 0.0}})
    r = client(svc).post("/api/chat", data={"session_id": "w1", "message": "write a readme for papertrail in a file", "agent": "docs"})
    data = r.get_json()
    assert r.status_code == 200 and data["files"] == ["README.md"] and data["tools"][0]["name"] == "save_file"
    listing = client(svc).get("/api/workspace?session_id=w1").get_json()
    assert listing == {"files": ["README.md"]}
    dl = client(svc).get("/api/workspace/file?session_id=w1&path=README.md")
    assert dl.status_code == 200 and dl.data.decode() == "# Papertrail\nSearches notes."
    assert 'attachment; filename="README.md"' in dl.headers["Content-Disposition"] and dl.mimetype == "text/plain"


def test_workspace_downloads_cannot_escape_the_session_or_reach_other_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_DOCS", "k")
    svc, api = make_service(tmp_path, api=FakeAPI(['<tool>{"name": "save_file", "args": {"path": "secret.txt", "content": "mine"}}</tool>', "done"]),
                            scheduler={"api_min_interval": {"gemini": 0.0}})
    c = client(svc)
    c.post("/api/chat", data={"session_id": "owner", "message": "save a file called secret please", "agent": "docs"})
    send(svc, "hello there", route="api", sid="other")
    for path in ("../owner/secret.txt", "/etc/passwd", "..\\..\\x", "C:\\x"):
        assert c.get("/api/workspace/file", query_string={"session_id": "other", "path": path}).status_code == 400
    assert c.get("/api/workspace/file", query_string={"session_id": "other", "path": "secret.txt"}).status_code == 400   # not theirs
    assert c.get("/api/workspace?session_id=nobody").status_code == 404 and c.get("/api/workspace?session_id=bad id!").status_code == 400
    assert c.get("/api/workspace?session_id=other").get_json() == {"files": []}


def test_tools_still_run_only_for_the_tools_the_current_agent_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_CODING", "k")
    call = '<tool>{"name": "save_file", "args": {"path": "x.txt", "content": "y"}}</tool>'
    svc, _ = make_service(tmp_path, api=FakeAPI([call, "ok"]), scheduler={"api_min_interval": {"gemini": 0.0}})
    out = svc.send("s", "", "", None, "save this as a file please 1", [], agent="coding")
    assert out.files == [] and out.tools == []                  # the coding agent has no save_file: it is not offered, so not even tried
    assert all(e["op"] != "file_write" for e in svc.os_kernel.audit)
