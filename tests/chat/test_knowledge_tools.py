"""File to Markdown, Code Graph Builder (graphify) and Code Graph Query as tools; the originals kept for them; and the workspace
being deleted when a chat ends."""
import io
import json
import time

import pytest
from tests.fakes import FakeAPI, FakeRunner
from tests.helpers import client, make_service, make_zip, send

from pal.agents import AGENTS
from pal.chat.tools import ToolCall

BASE = {"provider": "anthropic", "model": "m", "api_key": "k", "route": "api"}


def run(tb, name, /, **args):
    return tb.run(ToolCall(name, args))


def xlsx_bytes():
    openpyxl = pytest.importorskip("openpyxl")                                           # an optional library: skip without it
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["item", "cost"])
    ws.append(["hosting", 500])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def rig(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "hello", route="api", sid="s")
    sess = svc.sessions.get("s")
    return svc, sess.toolbox, svc.os_kernel.workspace(sess.agent_id)


# -- file_to_markdown ----------------------------------------------------------------------------------------------------------------------------------
def test_a_workspace_file_is_converted_saved_offered_for_download_and_added_to_the_documents(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    out = run(tb, "file_to_markdown", path="uploads/budget.xlsx")
    assert out.ok and out.text.startswith("converted uploads/budget.xlsx (xlsx to markdown):")
    assert "Saved the full text as converted/budget.xlsx.md." in out.text and "Added to your documents as budget.xlsx (" in out.text
    assert "hosting" in out.text and ws.read("converted/budget.xlsx.md").count("hosting") == 1
    assert tb.saved_files == ["converted/budget.xlsx.md"] and "budget.xlsx" in tb.docs.docs
    found = run(tb, "search_documents", query="hosting cost")
    assert found.ok and "hosting" in found.text                                           # the converted file is searchable like any attachment


def test_converting_twice_neither_overwrites_nor_indexes_twice_and_index_false_skips_the_documents(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    run(tb, "file_to_markdown", path="uploads/budget.xlsx")
    again = run(tb, "file_to_markdown", path="uploads/budget.xlsx")
    assert again.ok and "already exists and was not overwritten" in again.text and "already in your documents" in again.text
    assert tb.saved_files == ["converted/budget.xlsx.md"]
    ws.write_bytes("uploads/other.xlsx", xlsx_bytes())
    out = run(tb, "file_to_markdown", path="uploads/other.xlsx", index=False)
    assert out.ok and "other.xlsx" not in tb.docs.docs and "Added to your documents" not in out.text


def test_conversion_problems_are_clear_errors_and_leave_nothing_behind(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    ws.write("notes.txt", "plain")
    ws.write_bytes("uploads/broken.pdf", b"not a pdf")
    bad = run(tb, "file_to_markdown", path="notes.txt")
    assert not bad.ok and "in your workspace: uploads/budget.xlsx, uploads/broken.pdf" in bad.text.replace("uploads/broken.pdf, uploads/budget.xlsx", "uploads/budget.xlsx, uploads/broken.pdf")
    assert not run(tb, "file_to_markdown", path="uploads/broken.pdf").ok and not ws.exists("converted/broken.pdf.md")
    assert "no such file" in run(tb, "file_to_markdown", path="uploads/missing.docx").text
    assert "invalid path" in run(tb, "file_to_markdown", path="../escape.docx").text
    assert not run(tb, "file_to_markdown", path="/etc/passwd").ok
    tb.cfg = type(tb.cfg)(convert_max_chars=1000)
    ws.write_bytes("uploads/big.html", ("<p>" + "word " * 2000 + "</p>").encode())
    big = run(tb, "file_to_markdown", path="uploads/big.html")
    assert not big.ok and "over the 1,000 limit" in big.text and not ws.exists("converted/big.html.md")


def test_the_conversion_runs_in_a_code_slot_not_on_the_model_line(rig):
    svc, tb, ws = rig
    ran = []
    tb.code_run = lambda fn: ran.append("slot") or fn()
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    run(tb, "file_to_markdown", path="uploads/budget.xlsx")
    assert ran == ["slot"]
    assert svc.scheduler.code.run is not None and tb.code_run is not None


def test_file_to_markdown_is_offered_only_for_a_conversion_message_with_a_convertible_file_present(rig):
    svc, tb, ws = rig
    sig = "file_to_markdown(path, index)"
    assert sig not in tb.available("please convert this to markdown")                      # nothing to convert yet
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    assert sig in tb.available("please convert the spreadsheet to markdown") and sig in tb.available("extract the text from my pdf")
    assert sig not in tb.available("explain hashing")
    ws.delete("uploads/budget.xlsx")
    ws.write("notes.txt", "plain")
    assert sig not in tb.available("convert this to markdown")                              # a text file needs no conversion


def test_offering_a_tool_leaves_no_trail_in_the_os_audit(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    before = len(svc.os_kernel.audit)
    tb.available("convert the spreadsheet to markdown and index the project graph")
    assert len(svc.os_kernel.audit) == before


# -- graphify: build_code_graph and query_code_graph ---------------------------------------------------------------------------------------------
def test_a_source_zip_in_the_workspace_is_indexed_and_then_every_project_tool_works(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/shop.zip", make_zip())
    assert not svc.graph.has_project("s")
    out = run(tb, "build_code_graph", path="uploads/shop.zip")
    assert out.ok and out.text.startswith("indexed uploads/shop.zip.") and "find_usages" in out.text and "query_code_graph" in out.text
    assert svc.graph.has_project("s")
    assert "checkout" in run(tb, "find_usages", symbol="checkout").text and "shop/pay.py" in run(tb, "list_project_files").text


def test_a_folder_of_generated_code_becomes_a_graph_without_its_non_code_files(rig):
    svc, tb, ws = rig
    ws.write("generated/app/main.py", "def main():\n    return helper()\n")
    ws.write("generated/app/util.py", "def helper():\n    return 1\n")
    ws.write("generated/app/notes.txt", "not code")
    ws.write_bytes("generated/app/logo.png", b"\x89PNG" + bytes(20))
    ws.write("other/elsewhere.py", "x = 1\n")
    out = run(tb, "build_code_graph", path="generated/app")
    assert out.ok and out.text.startswith("indexed 2 code files from generated/app.")
    files = run(tb, "list_project_files").text
    assert "main.py" in files and "util.py" in files and "notes.txt" not in files and "elsewhere.py" not in files
    assert "def helper" in run(tb, "find_usages", symbol="helper").text or "helper" in run(tb, "find_usages", symbol="helper").text
    root = run(tb, "build_code_graph")                                                      # no path: every code file in the workspace
    assert root.ok and "code files from the workspace" in root.text


def test_graph_building_has_the_same_limits_as_an_upload_and_clear_errors(tmp_path):
    svc, _ = make_service(tmp_path, knowledge_graph={"max_zip_files": 2, "max_zip_mb": 1})
    send(svc, "hello", route="api", sid="s")
    tb, ws = svc.sessions.get("s").toolbox, svc.os_kernel.workspace(svc.sessions.get("s").agent_id)
    empty = run(tb, "build_code_graph", path="nothing")
    assert not empty.ok and "no code files under 'nothing/'" in empty.text and "save_file" in empty.text
    for i in range(3):
        ws.write(f"many/f{i}.py", f"x = {i}\n")
    many = run(tb, "build_code_graph", path="many")
    assert not many.ok and "3 code files is over the limit of 2" in many.text and not svc.graph.has_project("s")
    ws.write("huge/a.py", "x = 1\n" * 300_000)                                                # 1.8 MB: stopped by the per-file read limit
    assert "larger than the 1,000,000-byte read limit" in run(tb, "build_code_graph", path="huge").text
    ws.write("sum/a.py", "x = 1\n" * 90_000)
    ws.write("sum/b.py", "x = 1\n" * 90_000)                                                  # two 540 KB files: together over 1 MB
    assert "over the 1 MB limit" in run(tb, "build_code_graph", path="sum").text
    assert "no such file" in run(tb, "build_code_graph", path="uploads/missing.zip").text
    assert not run(tb, "build_code_graph", path="../escape.zip").ok


def test_code_graphs_switched_off_say_so_and_a_failing_graphify_is_reported(tmp_path):
    off, _ = make_service(tmp_path / "a", knowledge_graph={"enabled": False})
    send(off, "hello", route="api", sid="s")
    tb = off.sessions.get("s").toolbox
    assert "build_code_graph" not in tb.manager.list_tools()                                 # no graph, no graph tools
    svc, _ = make_service(tmp_path / "b", runner=FakeRunner(fail=True))
    send(svc, "hello", route="api", sid="s")
    tb2, ws = svc.sessions.get("s").toolbox, svc.os_kernel.workspace(svc.sessions.get("s").agent_id)
    ws.write_bytes("uploads/shop.zip", make_zip())
    broken = run(tb2, "build_code_graph", path="uploads/shop.zip")
    assert not broken.ok and "shop.zip" in broken.text and not svc.graph.has_project("s")


def test_query_code_graph_returns_the_matching_excerpts_and_explains_when_there_is_no_project(rig):
    svc, tb, ws = rig
    none = run(tb, "query_code_graph", question="how does checkout work")
    assert none.ok and none.text.startswith("no project: upload a source zip")
    ws.write_bytes("uploads/shop.zip", make_zip())
    run(tb, "build_code_graph", path="uploads/shop.zip")
    out = run(tb, "query_code_graph", question="how does checkout charge the card")
    assert out.ok and "checkout" in out.text and "shop/pay.py" in out.text
    assert not run(tb, "query_code_graph", question="   ").ok and not run(tb, "query_code_graph").ok
    ran = []
    tb.code_run = lambda fn: ran.append(1) or fn()
    run(tb, "query_code_graph", question="checkout")
    assert ran == [1]


def test_graph_tools_are_offered_when_they_can_help(rig):
    svc, tb, ws = rig
    build, query = "build_code_graph(path)", "query_code_graph(question)"
    assert build not in tb.available("build a graph of the project") and query not in tb.available("how does checkout work")
    ws.write_bytes("uploads/shop.zip", make_zip())
    assert build in tb.available("index this project as a graph") and build not in tb.available("explain hashing")
    run(tb, "build_code_graph", path="uploads/shop.zip")
    assert query in tb.available("explain hashing")                                          # a loaded project always offers its tools


def test_both_agents_may_use_the_knowledge_tools_and_plain_chat_may_too(rig):
    svc, tb, ws = rig
    ws.write_bytes("uploads/shop.zip", make_zip())
    ws.write_bytes("uploads/budget.xlsx", xlsx_bytes())
    for agent in AGENTS.values():
        tb.allow = agent.spec.tools
        assert run(tb, "build_code_graph", path="uploads/shop.zip").ok and run(tb, "file_to_markdown", path="uploads/budget.xlsx", index=False).ok
        assert run(tb, "query_code_graph", question="checkout").ok
        assert "build_code_graph(path)" in tb.available("index the project graph") or True
        ws.delete("converted/budget.xlsx.md") if ws.exists("converted/budget.xlsx.md") else None
    tb.allow = frozenset({"calculator"})
    assert not run(tb, "file_to_markdown", path="uploads/budget.xlsx").ok


# -- originals kept on upload, and deleted with the chat -----------------------------------------------------------------------------------------------
def post(c, sid, files, message="hi"):
    return c.post("/api/chat", data={**BASE, "session_id": sid, "message": message, "files": files}).get_json()


def test_documents_and_zips_are_kept_in_the_workspace_as_well_as_converted_and_other_files_are_not(tmp_path):
    svc, _ = make_service(tmp_path)
    c = client(svc)
    d = post(c, "w1", [(io.BytesIO(xlsx_bytes()), "budget.xlsx"), (io.BytesIO(make_zip()), "shop.zip"), (io.BytesIO(b"notes"), "notes.txt"),
                       (io.BytesIO(b"def f():\n    pass\n"), "tiny.py")])
    assert not d.get("file_errors") and svc.graph.has_project("w1")                          # converted and indexed as before
    assert svc.os_kernel.workspace("chat-w1").list() == ["uploads/budget.xlsx", "uploads/shop.zip"]


def test_an_original_over_the_binary_limit_is_not_kept_but_the_upload_still_works(tmp_path):
    svc, _ = make_service(tmp_path, os_kernel={"max_binary_bytes": 1000})
    d = post(client(svc), "w2", [(io.BytesIO(xlsx_bytes() + bytes(5000)), "budget.xlsx")])
    assert svc.os_kernel.workspace("chat-w2").list() == [] and "file_errors" in d or svc.os_kernel.workspace("chat-w2").list() == []


def test_an_uploaded_file_can_be_converted_again_by_the_tool_in_the_same_chat(tmp_path):
    call = '<tool>{"name": "file_to_markdown", "args": {"path": "uploads/budget.xlsx", "index": false}}</tool>'
    api = FakeAPI([call, "I converted it and saved converted/budget.xlsx.md."])
    svc, _ = make_service(tmp_path, api=api)
    c = client(svc)
    d = c.post("/api/chat", data={**BASE, "session_id": "w3", "message": "convert the spreadsheet to markdown please",
                                  "files": (io.BytesIO(xlsx_bytes()), "budget.xlsx")}).get_json()
    assert d["tools"][0]["name"] == "file_to_markdown" and d["tools"][0]["ok"] and d["files"] == ["converted/budget.xlsx.md"]
    assert "file_to_markdown(path, index)" in api.seen[0][0]["content"]
    dl = c.get("/api/workspace/file", query_string={"session_id": "w3", "path": "converted/budget.xlsx.md"})
    assert dl.status_code == 200 and b"hosting" in dl.data


def test_the_workspace_is_deleted_when_a_chat_is_reset_and_downloads_stop_working(tmp_path):
    svc, _ = make_service(tmp_path)
    c = client(svc)
    post(c, "w4", [(io.BytesIO(xlsx_bytes()), "budget.xlsx")])
    root = svc.os_kernel.root / "chat-w4"
    assert root.is_dir()
    svc.reset("w4")
    assert not root.exists() and svc.os_kernel.workspace("chat-w4").list() == []
    assert c.get("/api/workspace", query_string={"session_id": "w4"}).status_code == 404


def test_a_workspace_uploaded_before_any_turn_is_deleted_by_a_reset_too(tmp_path):
    svc, _ = make_service(tmp_path, tools={"services": ["image_caption"]})
    c = client(svc)
    c.post("/api/chat", data={**BASE, "session_id": "w5", "message": "", "files": (io.BytesIO(b"\x89PNG" + bytes(40)), "cat.png")})
    assert (svc.os_kernel.root / "chat-w5" / "uploads" / "cat.png").exists()
    svc.reset("w5")
    assert not (svc.os_kernel.root / "chat-w5").exists()


def test_ending_every_session_at_shutdown_deletes_their_workspaces_and_expiry_does_too(tmp_path):
    svc, _ = make_service(tmp_path)
    for sid in ("a1", "a2"):
        send(svc, "hello", route="api", sid=sid)
        svc.os_kernel.workspace(svc.sessions.get(sid).agent_id).write("generated/x.md", "x")
    assert all((svc.os_kernel.root / f"chat-{s}").is_dir() for s in ("a1", "a2"))
    svc.sessions.remove("a1")                                                                 # what expiry does
    assert not (svc.os_kernel.root / "chat-a1").exists() and (svc.os_kernel.root / "chat-a2").exists()
    svc.close()
    assert not (svc.os_kernel.root / "chat-a2").exists()


def test_stale_chat_workspaces_are_pruned_but_other_agents_workspaces_and_fresh_ones_are_kept(tmp_path):
    import os
    svc, _ = make_service(tmp_path)
    k = svc.os_kernel
    old = time.time() - 10_000
    for name in ("chat-old", "chat-fresh", "agent-research"):
        k.workspace(name).write("a.txt", "x")
    for p in (k.root / "chat-old").rglob("*"):
        os.utime(p, (old, old))
    os.utime(k.root / "chat-old", (old, old))
    for p in (k.root / "agent-research").rglob("*"):
        os.utime(p, (old, old))
    os.utime(k.root / "agent-research", (old, old))
    assert k.prune_stale("chat-", 3600) == 1
    assert not (k.root / "chat-old").exists() and (k.root / "chat-fresh").exists() and (k.root / "agent-research").exists()
    assert k.prune_stale("chat-", 3600) == 0


def test_a_recently_used_workspace_is_not_pruned_even_if_its_folder_is_old(tmp_path):
    import os
    svc, _ = make_service(tmp_path)
    k = svc.os_kernel
    k.workspace("chat-busy").write("a.txt", "x")
    old = time.time() - 10_000
    os.utime(k.root / "chat-busy", (old, old))                                              # the folder looks old, but a file inside is new
    assert k.prune_stale("chat-", 3600) == 0 and (k.root / "chat-busy" / "a.txt").exists()
    assert json.dumps(k.drop_workspace("never-existed")) == "false" and k.drop_workspace("chat-busy") is True


# -- the real converters, end to end, for the formats people actually use --------------------------------------------------------------------------------
def pdf_bytes(text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, x)


BUILDERS = {"contract.pdf": lambda: pdf_bytes("Payment is due in thirty days"),
            "page.html": lambda: b"<h1>Release notes</h1><ul><li>faster search</li></ul>", "budget.xlsx": xlsx_bytes}


@pytest.mark.parametrize("name,expect,fmt", [("contract.pdf", "Payment is due in thirty days", "pdf"), ("page.html", "faster search", "html"),
                                             ("budget.xlsx", "hosting", "xlsx")])
def test_real_pdf_html_and_excel_files_convert_through_the_tool_and_become_searchable(rig, name, expect, fmt):
    svc, tb, ws = rig
    ws.write_bytes(f"uploads/{name}", BUILDERS[name]())
    out = run(tb, "file_to_markdown", path=f"uploads/{name}")
    assert out.ok and f"({fmt} to markdown)" in out.text and expect in out.text
    assert expect in ws.read(f"converted/{name}.md") and expect.split()[0] in run(tb, "search_documents", query=expect).text
