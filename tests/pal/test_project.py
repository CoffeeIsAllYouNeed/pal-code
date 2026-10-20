"""Setting up the folder when pal cli / pal web starts: documents to markdown, the project map, the code graph, and the local model."""
import io
import json
import zipfile
from pathlib import Path

import pytest

from pal.terminal import project as pj
from pal.terminal.project import Project, ProjectGraph, build_map, detect_tests, ignored, outline, scan
from pal.tokens import estimate_tokens


def make_docx(paragraphs):
    """A minimal Word file: (style, text) paragraphs."""
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    style = '<w:pStyle w:val="{}"/>'
    body = "".join(f"<w:p><w:pPr>{style.format(s) if s else ''}</w:pPr><w:r><w:t>{t}</w:t></w:r></w:p>" for s, t in paragraphs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<?xml version="1.0"?><w:document {w}><w:body>{body}</w:body></w:document>')
        z.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


def make_pdf(text):
    """A one-page PDF with a text layer, without any PDF library."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % x for x in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "shop"
    (f / "app").mkdir(parents=True)
    (f / "app" / "cart.py").write_text("class Cart:\n    def __init__(self):\n        self.items = []\n\n    def add(self, item, qty=1):\n        pass\n\n\ndef total(cart):\n    return 0\n")
    (f / "web").mkdir()
    (f / "web" / "api.ts").write_text("export function getCart(id: string) { return fetch(id) }\nexport const price = (n: number) => n * 2\nclass Store {}\n")
    (f / "node_modules" / "x").mkdir(parents=True)
    (f / "node_modules" / "x" / "index.js").write_text("module.exports = 1")
    (f / ".env").write_text("KEY=sk-abcdefghijklmnopqrstuvwxyz123456")
    (f / "secret.pem").write_text("-----BEGIN PRIVATE KEY-----")
    (f / "docs").mkdir()
    (f / "docs" / "spec.docx").write_bytes(make_docx([("Heading1", "Checkout spec"), ("", "Lock the cart after payment."), ("ListParagraph", "card"), ("ListParagraph", "upi")]))
    (f / "docs" / "terms.pdf").write_bytes(make_pdf("Refunds within 30 days"))
    (f / "README.md").write_text("# Shop\nA small shop.")
    (f / "tests").mkdir()
    (f / "tests" / "test_cart.py").write_text("def test_cart():\n    assert True\n")
    return f


def fake_runner(calls):
    def run(args, cwd, timeout=60):
        calls.append(args)
        if args[0] == "update":
            out = Path(cwd) / "src" / "graphify-out"
            out.mkdir(parents=True, exist_ok=True)
            files = [p for p in (Path(cwd) / "src").rglob("*") if p.is_file() and "graphify-out" not in p.parts]
            (out / "graph.json").write_text(json.dumps({"nodes": [{"id": p.name} for p in files], "links": []}))
            return 0, "", ""
        if args[0] == "god-nodes":
            return 0, json.dumps([{"label": "Cart"}]), ""
        return 1, "", "unknown"
    return run


def test_scan_leaves_out_dependencies_hidden_files_secrets_and_palignore(folder):
    (folder / ".palignore").write_text("# generated\nweb/\n*.pdf\n")
    found = scan(folder)
    assert set(found.code) == {"app/cart.py", "tests/test_cart.py"}
    assert set(found.docs) == {"docs/spec.docx", "README.md"}
    assert ignored("build/out.js", ["build/"]) is True and ignored("src/a.js", ["build/"]) is False


def test_the_map_lists_classes_and_functions_for_python_and_other_languages(folder):
    assert outline("app/cart.py", (folder / "app/cart.py").read_text()) == ["class Cart", "  def __init__()", "  def add(item, qty)", "def total(cart)"]
    assert outline("web/api.ts", (folder / "web/api.ts").read_text()) == ["function getCart", "const price", "class Store"]
    text = build_map(folder, ["app/cart.py", "web/api.ts"])
    assert "## app/cart.py (11 lines)" in text and "- def total(cart)" in text and estimate_tokens(text) < 200


def test_documents_become_markdown_files_outside_the_folder(folder, tmp_path, monkeypatch):
    import pal.input_converters.components.pandoc_converter as pc
    monkeypatch.setattr(pc, "available", lambda: False)            # no pandoc on this laptop: Word files still convert
    p = Project(folder, home=tmp_path / "home", runner=fake_runner([]))
    r = p.prepare(graph_in_background=False)
    conv = {d.path: d for d in r.converted}
    assert set(conv) == {"docs/spec.docx", "docs/terms.pdf"}
    md = Path(conv["docs/spec.docx"].markdown).read_text()
    assert md.startswith("# Checkout spec") and "- card\n- upi" in md and "Refunds within 30 days" in Path(conv["docs/terms.pdf"].markdown).read_text()
    assert str(tmp_path / "home") in conv["docs/spec.docx"].markdown and not list(folder.rglob("*.docx.md"))       # nothing written in the folder
    names = dict(p.documents())
    assert {"docs/spec.docx", "docs/terms.pdf", "README.md", "project-map.md"} <= set(names)
    assert r.tests == "python -m pytest -q" and r.languages[0] == "Python" and r.code_files == 3


def test_the_second_start_redoes_only_what_changed(folder, tmp_path, monkeypatch):
    calls = []
    p = Project(folder, home=tmp_path / "home", runner=fake_runner(calls))
    p.prepare(graph_in_background=False)
    import pal.input_converters as ic
    converted = []
    real = ic.convert_to_markdown
    monkeypatch.setattr(ic, "convert_to_markdown", lambda name, raw: converted.append(name) or real(name, raw))
    r = Project(folder, home=tmp_path / "home", runner=fake_runner(calls)).prepare(graph_in_background=False)
    assert converted == [] and [c[0] for c in calls].count("update") == 1 and r.graph.startswith("ready")
    (folder / "app" / "cart.py").write_text("def total(cart):\n    return 1\n")
    (folder / "docs" / "terms.pdf").unlink()
    r = Project(folder, home=tmp_path / "home", runner=fake_runner(calls)).prepare(graph_in_background=False)
    assert [c[0] for c in calls].count("update") == 2 and "docs/terms.pdf" not in {d.path for d in r.docs}
    assert not list((tmp_path / "home" / "projects").glob("*/markdown/docs/terms.pdf.md"))


def test_the_folder_graph_serves_every_session_and_an_upload_keeps_its_own(folder, tmp_path):
    p = Project(folder, home=tmp_path / "home", runner=fake_runner([]))
    r = p.prepare(graph_in_background=False)
    assert r.graph.startswith("ready") and isinstance(p.graph, ProjectGraph)
    assert p.graph.has_project("any-session") and p.graph.has_project("another")
    p.graph.reset("any-session")                                   # a session ending never deletes the folder's graph
    assert p.graph.project_ws.exists()
    assert (p.root / "graph" / "project" / "src" / "app" / "cart.py").is_file()
    assert not (folder / "graphify-out").exists()


def test_without_graphify_the_rest_still_works(folder, tmp_path, monkeypatch):
    import pal.knowledge_graph.components.builder as b
    monkeypatch.setattr(b, "graphify_available", lambda: False)
    p = Project(folder, home=tmp_path / "home")
    r = p.prepare(graph_in_background=False)
    assert r.graph == "needs graphify" and r.map_tokens > 0 and r.converted


def test_the_test_command_is_found_from_the_project_files(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}')
    assert detect_tests(tmp_path) == "npm test"
    (tmp_path / "package.json").unlink()
    (tmp_path / "go.mod").write_text("module x")
    assert detect_tests(tmp_path) == "go test ./..."


def test_a_session_gets_the_brief_once_and_matching_document_passages(folder, tmp_path):
    from pal.terminal.clients import PalClient
    from pal.terminal.session import PalSession
    from tests.pal.fakes import ScriptedCore
    p = Project(folder, home=tmp_path / "home", runner=fake_runner([]))
    p.prepare(graph_in_background=False)
    seen = []

    class Core(ScriptedCore):
        def chat(self, syscall):
            seen.append(syscall.messages[-1]["content"])
            from pal.llm_wrapper.components.base import LLMResponse
            return LLMResponse("ok", prompt_tokens=10, completion_tokens=1)
    client = PalClient(Core(), None, frozenset(), home=tmp_path / "engine", local=False, project=p)
    s = PalSession(client, "m", "local", folder, "ask")
    s.project, s.brief = p, p.brief()
    try:
        s.ask("What does the checkout spec say about the cart after payment?")
        s.ask("and refunds?")
    finally:
        client.close()
    assert "[Project: shop" in seen[0] and "[Project: shop" not in seen[1]
    assert "Lock the cart after payment" in seen[0]
    assert "Refunds within 30 days" in seen[1]


def test_web_and_cli_skip_the_folder_setup_when_told(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from pal.terminal.cli import prepare_folder
    from pal.terminal.ui import make_console
    monkeypatch.setenv("PAL_SKIP_PROJECT", "1")
    assert prepare_folder(SimpleNamespace(dir=str(tmp_path), host=None), make_console(), core_factory=object()) == ("", None)


def test_prepare_folder_sets_up_the_folder(monkeypatch, tmp_path, folder):
    from types import SimpleNamespace

    from pal.terminal.cli import prepare_folder
    from pal.terminal.ui import make_console
    monkeypatch.delenv("PAL_SKIP_PROJECT")
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "home"))
    import pal.knowledge_graph.components.builder as b
    monkeypatch.setattr(b, "default_runner", fake_runner([]))
    monkeypatch.setattr(pj.KnowledgeGraph, "available", lambda self: True)
    local, project = prepare_folder(SimpleNamespace(dir=str(folder), host=None), make_console(), core_factory=object())
    project.wait(10)
    assert local == "" and project.report.code_files == 3 and project.report.graph.startswith("ready")


# -- the local model ------------------------------------------------------------------------------------------------------------
def _hw(spec):
    from pal.terminal.hardware import from_spec
    return from_spec(spec)


def test_no_gpu_means_no_local_model_to_prepare():
    from pal.terminal.setup import ensure_local
    from pal.terminal.ui import make_console
    st = ensure_local(make_console(), _hw("cpu:16"), info=lambda h: pytest.fail("must not look for Ollama"))
    assert st.state == "off"


def test_ollama_is_installed_after_asking_then_started_and_models_are_offered(monkeypatch, tmp_path):
    from pal.terminal import backend
    from pal.terminal.setup import ensure_local
    from pal.terminal.ui import make_console
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    state = {"installed": False, "running": False, "models": []}
    asked, done = [], []
    info = lambda host: {"ok": state["running"], "models": [{"name": m} for m in state["models"]]}  # noqa: E731

    def setup(console, hw, host, ask):
        done.append("setup")
        state["models"] = ["qwen2.5-coder:7b", "qwen2.5:1.5b"]
        backend.save_prefs(provider="local", model="qwen2.5-coder:7b", helper_model="qwen2.5:1.5b", setup_done="1")
    st = ensure_local(make_console(), _hw("gpu:8"), ask=lambda q: asked.append(q) or True,
                      installer=lambda console, system: state.update(installed=True), starter=lambda host: state.update(running=True) or True,
                      info=info, setup=setup, ollama=lambda: state["installed"])
    assert "Install it now?" in asked[0] and done == ["setup"]
    assert st.state == "ready" and "qwen2.5-coder:7b" in st.detail and "helper qwen2.5:1.5b" in st.detail
    st = ensure_local(make_console(), _hw("gpu:8"), ask=lambda q: pytest.fail("nothing to ask"), info=info, setup=lambda *a, **k: pytest.fail("no setup"),
                      ollama=lambda: True)
    assert st.state == "ready"


def test_declining_the_install_leaves_a_clear_note():
    from pal.terminal.setup import ensure_local
    from pal.terminal.ui import make_console
    st = ensure_local(make_console(), _hw("gpu:8"), ask=lambda q: False, ollama=lambda: False, info=lambda h: {"ok": False})
    assert st.state == "missing" and "ollama.com" in st.detail


def test_a_server_that_does_not_start_is_reported(monkeypatch, tmp_path):
    from pal.terminal.setup import ensure_local, start_ollama
    from pal.terminal.ui import make_console
    st = ensure_local(make_console(), _hw("gpu:8"), ollama=lambda: True, info=lambda h: {"ok": False}, starter=lambda h: False)
    assert st.state == "stopped" and "ollama serve" in st.detail
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ollama")
    started = []
    ok = start_ollama("http://localhost:11434", wait_s=1, info=lambda h: {"ok": len(started) > 0}, popen=lambda cmd, **kw: started.append(cmd), sleep=lambda s: None)
    assert ok and started == [["/usr/bin/ollama", "serve"]]


def test_the_install_command_fits_the_computer(monkeypatch):
    from pal.terminal import setup as st
    monkeypatch.setattr(st.shutil, "which", lambda name: "/x/" + name)
    assert st.install_command("Linux")[-1].startswith("curl -fsSL https://ollama.com/install.sh")
    assert st.install_command("Darwin") == ["brew", "install", "ollama"]
    assert st.install_command("Windows")[:4] == ["winget", "install", "-e", "--id"]
    monkeypatch.setattr(st.shutil, "which", lambda name: None)
    assert st.install_command("Darwin") is None


def test_real_graphify_builds_the_folder_graph(folder, tmp_path):
    from pal.knowledge_graph.components.builder import graphify_available
    if not graphify_available():
        pytest.skip("graphify is not installed")
    p = Project(folder, home=tmp_path / "home")
    r = p.prepare(graph_in_background=False)
    assert r.graph.startswith("ready"), r.graph
    block = p.graph.context_block("s", "how is the cart total computed", estimate_tokens)
    assert block is not None and "cart" in block.text.lower()


def test_pdf_helper_makes_a_readable_pdf(tmp_path):
    from pypdf import PdfReader
    (tmp_path / "x.pdf").write_bytes(make_pdf("hello pdf"))
    assert "hello pdf" in PdfReader(str(tmp_path / "x.pdf")).pages[0].extract_text()
