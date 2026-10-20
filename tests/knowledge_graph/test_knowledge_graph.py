import io

import pytest
from tests.fakes import FakeRunner, RouterLocal
from tests.helpers import client, make_service, make_zip
from tests.helpers import send as _send

from pal.knowledge_graph import GRAPH_NOTICE, GraphError, KnowledgeGraph
from pal.knowledge_graph.components.retriever import parse_nodes
from pal.knowledge_graph.components.workspace import Workspace

try:
    import graphify  # noqa: F401
    HAVE_GRAPHIFY = True
except ImportError:
    HAVE_GRAPHIFY = False
needs_graphify = pytest.mark.skipif(not HAVE_GRAPHIFY, reason="graphify not installed")

@pytest.fixture
def gc(tmp_path):
    return KnowledgeGraph({"dir": str(tmp_path), "budget_tokens": 800}, runner=FakeRunner())


# -- workspace safety ----------------------------------------------------------------------
def test_zip_keeps_only_code_and_keeps_paths_as_in_the_archive(tmp_path):
    ws = Workspace(tmp_path, "s")
    n = ws.add_zip(make_zip(wrapper="proj/", extra={"proj/README.md": "docs", "proj/img.png": "x",
                                                      "proj/node_modules/x.js": "1", "proj/.git/a.py": "1"}), 100, 10**7)
    assert n == 3
    assert sorted(p.relative_to(ws.src).as_posix() for p in ws.source_files()) == [
        "proj/shop/cart.py", "proj/shop/pay.py", "proj/shop/unrelated.py"]


@pytest.mark.parametrize("bad", ["../evil.py", "/abs/evil.py", "a/../../evil.py"])
def test_zip_path_traversal_is_ignored_or_rejected(tmp_path, bad):
    ws = Workspace(tmp_path, "s")
    try:
        ws.add_zip(make_zip(extra={bad: "x=1"}), 100, 10**7)
    except GraphError:
        pass
    assert not (tmp_path / "evil.py").exists() and not (tmp_path.parent / "evil.py").exists()


def test_names_windows_cannot_store_are_skipped_on_every_os(tmp_path):
    ws = Workspace(tmp_path, "s")
    n = ws.add_zip(make_zip(extra={"a/con.py": "x = 1", "a/b:c.py": "x = 2", "a/what?.py": "x = 3",
                                   "a/trail .py": "x = 4", "a/ok.py": "x = 5"}), 100, 10**7)
    names = sorted(p.name for p in ws.source_files())
    assert "ok.py" in names and not {"con.py", "b:c.py", "what?.py"} & set(names) and n == len(names)


def test_zip_limits_and_errors(tmp_path):
    ws = Workspace(tmp_path, "s")
    with pytest.raises(GraphError, match="not a valid zip"):
        ws.add_zip(b"nope", 10, 10**7)
    with pytest.raises(GraphError, match="no source code"):
        ws.add_zip(make_zip({"a.md": "x"}), 10, 10**7)
    with pytest.raises(GraphError, match="more than 2"):
        ws.add_zip(make_zip({f"f{i}.py": "x=1" for i in range(5)}), 2, 10**7)
    with pytest.raises(GraphError, match="over"):
        ws.add_zip(make_zip({"a.py": "x=1\n" * 500}), 10, 100)


def test_session_id_cannot_escape_the_graph_dir(tmp_path):
    assert Workspace(tmp_path, "../../etc").dir.parent == tmp_path


# -- ingest / retrieve -------------------------------------------------------------------------
def test_ingest_returns_a_small_summary_not_the_source(gc):
    att = gc.ingest("s", "shop.zip", make_zip())
    assert att["name"] == "shop.zip" and "indexed as a code graph" in att["note"]
    assert "3 source files" in att["text"] and "Cart" in att["text"] and "charge_card" not in att["text"]
    assert gc.has_project("s") and not gc.has_project("other")


def test_retrieve_returns_nodes_and_code_excerpts_within_budget(gc):
    gc.ingest("s", "shop.zip", make_zip())
    ex = gc.retrieve("s", "how is checkout paid", lambda t: max(1, len(t) // 4))
    assert "<knowledge_graph>" in ex.text and "checkout()  shop/pay.py:4" in ex.text
    assert "### checkout()" in ex.text and "amount = cart.total(prices)" in ex.text
    assert "weather" not in ex.text and ex.excerpts == 1 and ex.tokens <= 800
    assert ex.project_tokens > 0


def test_excerpt_uses_the_python_function_boundary(gc):
    gc.ingest("s", "shop.zip", make_zip())
    text = gc.retrieve("s", "q", lambda t: len(t) // 4).text
    assert "return charge_card(amount)" in text and "def charge_card" not in text


def test_budget_drops_excerpts_before_the_node_map(tmp_path):
    g = KnowledgeGraph({"dir": str(tmp_path), "budget_tokens": 5}, runner=FakeRunner())
    g.ingest("s", "shop.zip", make_zip())
    ex = g.retrieve("s", "q", lambda t: len(t) // 4)
    assert ex.excerpts == 0 and "checkout()" in ex.text


def test_retrieve_is_none_without_project_or_when_graphify_finds_nothing(tmp_path):
    g = KnowledgeGraph({"dir": str(tmp_path)}, runner=FakeRunner(nodes=[]))
    assert g.retrieve("s", "q", len) is None
    g.ingest("s", "shop.zip", make_zip())
    assert g.retrieve("s", "q", len) is None


def test_graphify_failure_is_a_clear_error(tmp_path):
    g = KnowledgeGraph({"dir": str(tmp_path)}, runner=FakeRunner(fail=True))
    with pytest.raises(GraphError, match="shop.zip: graphify could not build"):
        g.ingest("s", "shop.zip", make_zip())


def test_wants_zip_and_big_code_files_only(gc):
    assert gc.wants("p.zip", b"x") and gc.wants("big.py", b"x" * 9000)
    assert not gc.wants("small.py", b"x = 1") and not gc.wants("notes.md", b"x" * 9000)
    assert not KnowledgeGraph({"enabled": False}, runner=FakeRunner()).wants("p.zip", b"x")


def test_reset_deletes_the_project(gc):
    gc.ingest("s", "shop.zip", make_zip())
    gc.reset("s")
    assert not gc.has_project("s")


def test_parse_nodes_handles_ranking_dedup_and_noise():
    out = ("Graph: g | x\n\n[!] TRUNCATED\n\nNODE a() [src=src/m.py loc=L3 community=X Y]\n"
           "NODE a() [src=src/m.py loc=L3 community=X Y]\nNODE b [src=src/n.py loc=L9]\n... (truncated)\n")
    assert parse_nodes(out) == [("a()", "src/m.py", 3), ("b", "src/n.py", 9)]


# -- the real graphify ---------------------------------------------------------------------------------
@needs_graphify
def test_real_graphify_indexes_and_retrieves_relevant_code(tmp_path):
    g = KnowledgeGraph({"dir": str(tmp_path), "budget_tokens": 1200})
    att = g.ingest("real", "shop.zip", make_zip())
    assert "3 source files" in att["text"]
    ex = g.retrieve("real", "how does checkout charge the card", lambda t: max(1, len(t) // 4))
    assert ex is not None and "checkout" in ex.text and "charge_card" in ex.text


# -- ChatService integration ---------------------------------------------------------------------------
def upload(svc, sid="session-1"):
    return svc.graph.ingest(sid, "shop.zip", make_zip())


def send(svc, msg, atts=None, sid="session-1"):
    return _send(svc, msg, atts, sid=sid)


def test_api_request_carries_excerpts_and_notice_but_history_does_not(tmp_path):
    svc, api = make_service(tmp_path)
    out = send(svc, "how does checkout work", [upload(svc)])
    msgs = api.seen[-1]
    assert GRAPH_NOTICE in msgs[0]["content"] and "<knowledge_graph>" in msgs[-1]["content"]
    assert out.graph["excerpts"] == 1 and out.graph["nodes"] == 1
    assert "[file: shop.zip | indexed as a code graph; source not included]" in msgs[-1]["content"]
    assert all("<knowledge_graph>" not in t.user for t in svc.sessions.get("session-1").turns)
    send(svc, "and charge_card?")
    later = api.seen[-1]
    assert sum("<knowledge_graph>" in m["content"] for m in later if m["role"] != "system") == 1  # only the new turn
    assert svc.metrics.chat_summary()["tokens_saved_estimate"]["code_graph"] > 0


def test_project_questions_skip_local_routing_even_if_the_classifier_says_local(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(route='{"route": "local", "reason": "easy"}'))
    upload(svc)
    out = send(svc, "what does checkout do")
    assert out.handled_by == "api" and api.seen


def test_no_project_means_no_graph_notice_or_excerpts(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(route='{"route": "api", "reason": "x"}'))
    out = send(svc, "explain monads")
    assert out.graph is None and GRAPH_NOTICE not in api.seen[-1][0]["content"]


def test_retrieval_failure_never_breaks_the_chat(tmp_path):
    svc, api = make_service(tmp_path)
    upload(svc)
    svc.graph.retrieve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert send(svc, "what is checkout").reply == "api reply"


def test_new_chat_deletes_the_indexed_project(tmp_path):
    svc, api = make_service(tmp_path)
    upload(svc)
    svc.reset("session-1")
    assert not svc.graph.has_project("session-1")


def test_read_source_tool_reads_only_indexed_source_files(tmp_path):
    svc, api = make_service(tmp_path)
    upload(svc)
    g = svc.graph
    body = g.read_source("session-1", "shop/pay.py", 1, 6)
    assert "def checkout" in body and body.startswith("[shop/pay.py lines 1-6")
    for bad in ("../../etc/passwd", "/etc/passwd", "shop/../../x", "shop/nope.py", "C:\\x.py", "graphify-out/graph.json"):
        with pytest.raises(Exception, match="no such file"):
            g.read_source("session-1", bad)


# -- web endpoint ------------------------------------------------------------------------------------------------
def test_flask_zip_upload_is_indexed_and_next_question_gets_excerpts(tmp_path):
    svc, api = make_service(tmp_path)
    c = client(svc)
    base = {"session_id": "web", "provider": "anthropic", "model": "m", "api_key": "k"}
    r = c.post("/api/chat", data={**base, "message": "understand this", "files": (io.BytesIO(make_zip()), "shop.zip")})
    d = r.get_json()
    assert r.status_code == 200 and not d["file_errors"] and d["graph"]["nodes"] == 1
    assert "source not included" in d["sent"] and "def checkout" not in d["sent"].split("<knowledge_graph>")[0]
    r = c.post("/api/chat", data={**base, "message": "how does checkout work"})
    assert "amount = cart.total(prices)" in r.get_json()["sent"]
    assert c.get("/api/config").get_json()["graph"]["enabled"] is True
    c.post("/api/reset", json={"session_id": "web"})
    assert not svc.graph.has_project("web")


def test_flask_bad_zip_is_a_file_error_not_a_crash(tmp_path):
    svc, api = make_service(tmp_path)
    r = client(svc).post("/api/chat", data={"session_id": "x", "message": "hi", "provider": "anthropic",
                                            "model": "m", "api_key": "k",
                                            "files": (io.BytesIO(b"not a zip"), "p.zip")})
    assert r.status_code == 200 and "not a valid zip" in r.get_json()["file_errors"][0]


# -- the Knowledge Graph's block for the context manager ---------------------------------------------------------------------------------
def test_context_block_gives_the_excerpts_the_saving_and_the_description_for_a_request(gc):
    from pal.knowledge_graph import GraphBlock
    gc.ingest("s", "shop.zip", make_zip())
    count = lambda t: max(1, len(t) // 4)  # noqa: E731
    block = gc.context_block("s", "how is checkout paid", count)
    ex = gc.retrieve("s", "how is checkout paid", count)
    assert isinstance(block, GraphBlock) and block.text == ex.text and block.saved_tokens == max(0, ex.project_tokens - ex.tokens)
    assert block.info == {"nodes": ex.nodes, "excerpts": ex.excerpts, "tokens": ex.tokens, "project_tokens": ex.project_tokens}


def test_context_block_is_none_without_a_project_or_a_question_and_never_raises_when_the_graph_fails(gc, monkeypatch):
    count = lambda t: len(t) // 4  # noqa: E731
    assert gc.context_block("nobody", "anything", count) is None
    gc.ingest("s", "shop.zip", make_zip())
    assert gc.context_block("s", "   ", count) is None
    monkeypatch.setattr(gc, "retrieve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("graphify exploded")))
    assert gc.context_block("s", "how is checkout paid", count) is None            # a failing graph must not fail the chat
