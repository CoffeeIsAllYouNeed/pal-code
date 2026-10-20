"""The RAG build experiment: a local model builds a small RAG engine, with and without Pal, and the report shows the tokens saved.

The experiment never simulates. These tests check its machinery (steps, reading replies, the safety filter, writing files, the check that the engine
runs, the numbers and the report) and, with the real LiteLLM against a fake Ollama server that exists only here, the whole run.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from experiments.demo import cli
from experiments.demo import rag_build as G
from experiments.demo.models import Call, Probe, Recorder

ROOT = Path(__file__).resolve().parents[2]

REFERENCE = {
    "chunker.py": '''def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
''',
    "embedder.py": '''import hashlib
import math
import re


def embed(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0
''',
    "store.py": '''from chunker import chunk_text
from embedder import cosine, embed


class VectorStore:
    def __init__(self):
        self.items = []

    def add(self, text: str) -> None:
        for chunk in chunk_text(text):
            self.items.append((embed(chunk), chunk))

    def search(self, query: str, k: int = 3) -> list[tuple[float, str]]:
        q = embed(query)
        return sorted(((cosine(q, v), c) for v, c in self.items), key=lambda t: -t[0])[:k]
''',
    "prompt.py": '''def build_prompt(question: str, chunks: list[str]) -> str:
    passages = "\\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    return f"Answer only from the passages.\\n\\n{passages}\\n\\nQuestion: {question}"
''',
    "rag.py": '''from prompt import build_prompt
from store import VectorStore


class RAG:
    def __init__(self):
        self.store = VectorStore()

    def add_text(self, text: str) -> None:
        self.store.add(text)

    def ask(self, question: str, k: int = 3) -> str:
        return build_prompt(question, [c for _, c in self.store.search(question, k)])
''',
    "cli.py": '''import argparse

from rag import RAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("question")
    args = parser.parse_args()
    engine = RAG()
    engine.add_text(Path(args.file).read_text())
    print(engine.ask(args.question))
''',
    "README.md": "# Mini RAG\n\nA tiny retrieval-augmented generation engine.\n\n```python\nfrom rag import RAG\n```\n",
    "test_rag.py": "import unittest\n\nfrom chunker import chunk_text\n\n\nclass T(unittest.TestCase):\n    def test_chunk(self):\n        self.assertEqual(chunk_text(''), [])\n",
    "demo.py": "from rag import RAG\n\nr = RAG()\nr.add_text('Paris is the capital of France.')\nprint(r.ask('capital of France?'))\n",
}


def fenced(name, body):
    return f"```{'markdown' if name.endswith('.md') else 'python'}\n{body}```"


# -- the steps --------------------------------------------------------------------------------------------------------------------------------
def test_the_build_has_nine_steps_in_a_sensible_order_and_each_size_takes_a_prefix():
    assert [s.file for s in G.STEPS] == ["chunker.py", "embedder.py", "store.py", "prompt.py", "rag.py", "cli.py", "README.md", "test_rag.py", "demo.py"]
    assert G.SIZES == {"small": 5, "big": 7, "max": 9} and len({s.file for s in G.STEPS}) == 9
    assert [s.file for s in G.STEPS[:G.SIZES["small"]]] == ["chunker.py", "embedder.py", "store.py", "prompt.py", "rag.py"]      # the five the engine needs to run


def test_every_step_message_names_the_step_and_file_and_shows_the_signature_the_next_files_depend_on():
    n = len(G.STEPS)
    first, later = G.step_message(1, n, G.STEPS[0]), G.step_message(3, n, G.STEPS[2])
    assert first.startswith("We are building a small retrieval-augmented generation (RAG) engine") and "Step 1 of 9: write the file `chunker.py`" in first
    assert "def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]" in first and "from rag import RAG" in first
    assert later.startswith("Continue building the same engine. Use the earlier files exactly as written") and "Step 3 of 9: write the file `store.py`" in later and "class VectorStore" in later
    assert "```markdown block" in G.step_message(7, n, G.STEPS[6]) and "```python block" in later
    assert all(f"`{s.file}`" in G.step_message(i, n, s) for i, s in enumerate(G.STEPS, 1))


def test_the_design_document_is_pasted_into_the_plain_clients_first_message_only():
    n, docs = len(G.STEPS), "# Design notes\n\nthe retrieval details live here"
    first, second = G.step_message(1, n, G.STEPS[0], docs), G.step_message(2, n, G.STEPS[1], docs)
    assert "The design documentation of LocalRAG, for reference:\n\n# Design notes" in first and docs not in second and docs not in G.step_message(1, n, G.STEPS[0])
    assert first.index("The design documentation") < first.index("Step 1 of 9") and "design reference is the documentation of an existing, much larger RAG engine" in first


# -- reading replies and the safety filter ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reply,file,expect", [
    ("Here:\n```python\ndef f():\n    return 1\n```\nDone", "chunker.py", "def f():\n    return 1\n"),
    ("```\nx = 1\n```", "chunker.py", "x = 1\n"), ("def f():\n    return 2", "chunker.py", "def f():\n    return 2\n"),
    ("```python\ndef broken(:\n```\n```python\nok = 1\n```", "chunker.py", "ok = 1\n"),
    ("```markdown\n# Title\n\ntext\n```", "README.md", "# Title\n\ntext\n"), ("# Title only, no fence", "README.md", "# Title only, no fence\n"),
    ("I cannot help with that.", "chunker.py", None), ("```python\ndef broken(:\n```", "chunker.py", None), ("", "README.md", None)])
def test_a_file_is_read_from_a_fenced_block_a_bare_reply_or_the_first_block_that_parses(reply, file, expect):
    assert G.extract_file(reply, file) == expect


@pytest.mark.parametrize("code,reason", [("import os\nos.getcwd()", "importing os is not allowed"), ("import subprocess", "importing subprocess"), ("import numpy as np", "importing numpy"),
                                         ("x = open('f').read()", "calling open"), ("eval('1')", "calling eval"), ("from pathlib import Path\nPath('x').unlink()", "using .unlink"),
                                         ("from pathlib import Path\nPath('x').write_text('a')", "using .write_text"), ("x = ().__class__", "using .__class__"), ("def f(:", "syntax error")])
def test_the_file_filter_refuses_writes_processes_network_and_dynamic_code(code, reason):
    assert reason in G.unsafe_file_reason(code)


def test_the_standard_library_the_engine_really_needs_is_allowed():
    for name, body in REFERENCE.items():
        if name.endswith(".py"):
            assert G.unsafe_file_reason(body) in ("", "importing pathlib is not allowed") or name == "cli.py", name
    assert G.unsafe_file_reason("import hashlib, math, re, argparse, unittest\nfrom pathlib import Path\nfrom store import VectorStore\n") == ""


# -- writing the files and checking the engine ------------------------------------------------------------------------------------------------
def arm_with(replies_by_file, files=None):
    arm = G.ArmRag("baseline", Recorder())
    for i, step in enumerate(G.STEPS[:len(files or G.STEPS)], 1):
        arm.replies.append((step, replies_by_file.get(step.file, "no code"), 1.0, i))
    return arm


def test_an_engine_built_from_correct_files_is_written_to_disk_and_runs_end_to_end(tmp_path):
    arm = arm_with({k: fenced(k, v) for k, v in REFERENCE.items()})
    G.write_arm(arm, tmp_path / "baseline")
    assert arm.written == list(REFERENCE) and arm.smoke_ran and arm.smoke_ok and "answered the test question correctly" in arm.smoke_detail
    assert (tmp_path / "baseline" / "rag" / "rag.py").read_text(encoding="utf-8") == REFERENCE["rag.py"]
    assert (tmp_path / "baseline" / "replies" / "05_rag.py.md").read_text(encoding="utf-8").startswith("```python") and (tmp_path / "baseline" / "check_output.txt").is_file()


def test_an_engine_that_runs_but_answers_wrongly_is_reported_as_not_working_with_the_reason(tmp_path):
    wrong = dict(REFERENCE, **{"rag.py": REFERENCE["rag.py"].replace("self.store.search(question, k)", "[]")})
    arm = arm_with({k: fenced(k, v) for k, v in wrong.items()}, G.STEPS[:5])
    G.write_arm(arm, tmp_path / "x")
    assert arm.smoke_ran and not arm.smoke_ok and arm.smoke_detail.startswith("AssertionError: Answer only from the passages")


def test_a_missing_file_a_broken_import_and_an_unsafe_file_each_stop_the_check_and_say_why(tmp_path):
    missing = arm_with({k: fenced(k, v) for k, v in REFERENCE.items() if k != "store.py"}, G.STEPS[:5])
    G.write_arm(missing, tmp_path / "a")
    assert not missing.smoke_ran and "no usable store.py" in missing.smoke_detail and missing.problems["store.py"] == "no usable code in the reply"
    unsafe = arm_with({k: fenced(k, v) for k, v in dict(REFERENCE, **{"chunker.py": "import os\n" + REFERENCE["chunker.py"]}).items()}, G.STEPS[:5])
    G.write_arm(unsafe, tmp_path / "b")
    assert not unsafe.smoke_ran and "chunker.py: importing os is not allowed" in unsafe.smoke_detail and not (tmp_path / "b" / "check_output.txt").exists()
    broken = arm_with({k: fenced(k, v) for k, v in dict(REFERENCE, **{"store.py": REFERENCE["store.py"].replace("from chunker import chunk_text", "from chunker import nothing_here")}).items()}, G.STEPS[:5])
    G.write_arm(broken, tmp_path / "c")
    assert broken.smoke_ran and not broken.smoke_ok and "ImportError" in broken.smoke_detail


def test_code_that_never_finishes_is_stopped_and_counted_as_not_working(tmp_path):
    looping = dict(REFERENCE, **{"embedder.py": "while True:\n    pass\n"})
    arm = arm_with({k: fenced(k, v) for k, v in looping.items()}, G.STEPS[:5])
    G.write_arm(arm, tmp_path / "x", timeout=3)
    assert arm.smoke_ran and not arm.smoke_ok and "did not finish within 3 seconds" in arm.smoke_detail


def test_the_check_runs_in_an_environment_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-seen")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    env = G.safe_env(tmp_path)
    assert "OPENAI_API_KEY" not in env and env["SYSTEMROOT"] == "C:\\Windows" and env["PYTHONPATH"] == str(tmp_path) and env["PYTHONUTF8"] == "1"


# -- the numbers and the report ----------------------------------------------------------------------------------------------------------------
def fake_arm(name, prompts, answers, seconds, helper=0, smoke_ok=True, files=5):
    rec, arm = Recorder(), G.ArmRag(name, Recorder())
    arm.recorder = rec
    for i, (p, a) in enumerate(zip(prompts, answers, strict=True), 1):
        rec.add(Call(i, "main", "m", p, a, seconds, 0.0, server_prompt=p))
        if helper:
            rec.add(Call(i, "helper", "m", helper, 10, 0.5, 0.0))
        arm.replies.append((G.STEPS[i - 1], "r", seconds, i))
    arm.smoke_ran, arm.smoke_ok, arm.smoke_detail = True, smoke_ok, "the engine answered the test question correctly" if smoke_ok else "AssertionError"
    arm.written = [s.file for s in G.STEPS[:files]]
    return arm


def meta(folder="/main/experiment_output/20261003-100000"):
    return {"started": "2026-10-03 10:00:00", "size": "small", "steps_count": 5, "main_model": "llama3.2:3b", "helper_model": "llama3.2:3b", "num_ctx": 12288,
            "max_output": 900, "seed": 7, "hot_turns": 6, "spec_tokens": 4481, "folder": folder, "notes": []}


def test_the_saving_is_what_the_plain_client_used_minus_what_pal_used_and_the_helpers_cost_is_shown():
    base = fake_arm("baseline", [1000, 2000, 3000, 4000, 5000], [400] * 5, 20.0)
    mat = fake_arm("pal", [1100, 1500, 1500, 1500, 1500], [400] * 5, 25.0, helper=300)
    s = G.summarize(meta(), base, mat)
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    assert b["main_tokens"] == 15000 + 2000 and m["main_tokens"] == 7100 + 2000 and b["prompt"] == 15000 and m["helper_tokens"] == 5 * 310
    assert sv["main_tokens"] == 17000 - 9100 == 7900 and sv["main_tokens_pct"] == 46.5 and sv["prompt_pct"] == 52.7 and sv["all_tokens_pct"] == 37.4
    assert sv["seconds_pct"] == -25.0 and [x["prompt"] for x in s["steps"]["baseline"]] == [1000, 2000, 3000, 4000, 5000] and s["steps"]["pal"][0]["helper"] == 310


def test_the_report_leads_with_the_tokens_without_and_with_pal_and_how_many_were_saved():
    base = fake_arm("baseline", [1000, 2000, 3000, 4000, 5000], [400] * 5, 20.0)
    mat = fake_arm("pal", [1100, 1500, 1500, 1500, 1500], [400] * 5, 25.0, helper=300, smoke_ok=False)
    s = G.summarize({**meta(), "notes": ["a note"]}, base, mat)
    md = G.render_markdown(s)
    assert "### Without Pal: 17,000 tokens.  With Pal: 9,100 tokens.  **Saved: 7,900 tokens (46.5%).**" in md
    for part in ("**REAL RUN, no simulation.**", "> **Note.** a note", "## The result", "## Step by step", "## The files it wrote", "## How far to trust this", "## Setup",
                 "Helper tokens (Pal's own small jobs, an extra cost)", "25.0% slower", "(also used as Pal's helper)", "it keeps the last 6 steps in full", "--no-spec", "design documentation of LocalRAG (4,481 tokens)", "resends it at every step"):
        assert part in md, part
    assert "| yes | no (AssertionError) |" in md and "SIMULATED" not in md and md.count("| `chunker.py`") == 1 and md.count("| `") == 5 and "/main/experiment_output/20261003-100000/baseline/rag/" in md
    term = G.render_terminal(s, "/main/experiment_output/x")
    flat = re.sub(r" +", " ", term)
    assert "WITHOUT Pal (plain chat client): 17,000 tokens" in flat and "WITH Pal: 9,100 tokens" in flat and "TOKENS SAVED: 7,900 (46.5%)" in flat
    assert "slower" in term and "Files written to: /main/experiment_output/x" in term and "37.4%" in term


def test_without_a_design_document_the_report_says_so_and_what_that_means_for_the_saving():
    s = G.summarize({**meta(), "spec_tokens": 0}, fake_arm("baseline", [1000, 2000, 3000, 4000, 5000], [400] * 5, 20.0), fake_arm("pal", [1100, 2100, 3100, 4100, 5100], [400] * 5, 25.0))
    md = G.render_markdown(s)
    assert "No design document was attached in this run" in md and "needs a long build to pay off" in md and "design documentation of LocalRAG (" not in md and s["savings"]["main_tokens_pct"] < 0


def test_pals_own_estimates_are_shown_as_estimates_and_labelled_as_such():
    s = G.summarize(meta(), fake_arm("baseline", [3000] * 5, [400] * 5, 20.0), fake_arm("pal", [1500] * 5, [400] * 5, 25.0))
    s["estimates"] = {"documents": 12000, "context_paging": 800, "input_savers": 150, "total": 12950}
    md = G.render_markdown(s)
    assert "Pal's own estimate of where it saved tokens" in md and "not a measurement" in md
    assert "| Design-document excerpts instead of the whole document | 12,000 |" in md and "| History paging | 800 |" in md and "| Trimming what was typed | 150 |" in md


def test_the_cache_note_appears_only_when_the_server_skipped_most_of_the_plain_clients_prompt():
    base = fake_arm("baseline", [4000] * 5, [400] * 5, 20.0)
    for c in base.recorder.calls:
        c.server_prompt = 500
    mat = fake_arm("pal", [1500] * 5, [400] * 5, 25.0)
    assert any("reused a cached start of the prompt" in n for n in G.summarize(meta(), base, mat)["notes"])
    assert not any("cached start" in n for n in G.summarize(meta(), fake_arm("baseline", [4000] * 5, [400] * 5, 20.0), mat)["notes"])


def test_the_context_window_grows_with_the_build_and_the_defaults_are_what_the_guide_says():
    assert G.needed_ctx(9, 900) % 1024 == 0 and G.needed_ctx(9, 900) > G.needed_ctx(5, 900) > 8000 and G.needed_ctx(5, 900, 4500) > G.needed_ctx(5, 900) + 5000
    a = G.parser().parse_args([])
    assert (a.size, a.hot_turns, a.helper_model, a.max_output, a.no_spec) == ("big", 6, "", 900, False) and a.out_dir == str(ROOT / "experiment_output")
    assert G.parser().parse_args(["--no-spec"]).no_spec is True
    with pytest.raises(SystemExit):
        G.parser().parse_args(["--simulate"])


def test_without_ollama_it_stops_with_instructions_and_never_simulates(tmp_path, capsys):
    code = cli.main(["rag", "--ollama-url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2 and "does not simulate" in err and "ollama serve" in err and "ollama pull llama3.2:3b" in err and not (tmp_path / "out").exists()


def test_a_missing_model_lists_what_you_have_and_a_missing_helper_falls_back_to_the_main_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(G, "probe_ollama", lambda url, wanted: Probe(False, ["llama3.2:3b"], "", ["qwen2.5:7b"]))
    assert G.main(["--out-dir", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "The model llama3.2:3b is not installed" in err and "Installed models: qwen2.5:7b" in err and "python run.py experiment rag --model qwen2.5:7b" in err
    monkeypatch.setattr(G, "probe_ollama", lambda url, wanted: Probe(False, ["tiny"], "", ["llama3.2:3b"]))
    monkeypatch.setattr(G, "build_real_core", lambda *a: (_ for _ in ()).throw(RuntimeError("stop: models chosen")))
    summary, err2 = G.run(G.parser().parse_args(["--helper-model", "tiny", "--out-dir", str(tmp_path / "p"), "--quiet"]), lambda *a: None)
    assert summary is None and "stop: models chosen" in err2


# -- the whole run, through the real LiteLLM, against a fake Ollama that exists only in these tests -----------------------------------------------
class FakeRag(BaseHTTPRequestHandler):
    log: list = []
    mode = {"quality": "good"}

    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"models": [{"name": "llama3.2:3b"}, {"name": "tiny:1b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeRag.log.append(body)
        if "messages" not in body:
            self._json({})
            return
        msgs = body["messages"]
        system, user = msgs[0]["content"], msgs[-1]["content"]
        m = re.search(r"<text>\n(.*)\n</text>", user, re.S)
        found = re.search(r"write the file `([\w.]+)`", user)
        if system.startswith("You route"):
            text = '{"route": "api", "reason": "x"}'
        elif system.startswith("You audit"):
            text = '{"ok": true, "lost": []}'
        elif m:
            text = f"<text>\n{m.group(1)}\n</text>"
        elif user == "Reply with the single word: ready":
            text = "ready"
        elif found:
            body_text = REFERENCE[found.group(1)] + "\n".join(f"# a long explanatory note number {i} about this file" for i in range(25))
            text = fenced(found.group(1), body_text + "\n")
        else:
            text = "I do not know."
        self._json({"model": "m", "message": {"role": "assistant", "content": text}, "done": True, "prompt_eval_count": sum(len(x["content"]) for x in msgs) // 4, "eval_count": len(text) // 4})


@pytest.fixture
def fake_rag():
    FakeRag.log = []
    s = ThreadingHTTPServer(("127.0.0.1", 0), FakeRag)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


@pytest.fixture
def real_litellm():
    pytest.importorskip("litellm")
    from pal.llm_wrapper.components import litellm_wrapper
    litellm_wrapper._module = None
    yield litellm_wrapper._import_litellm()
    litellm_wrapper._module = None


def test_a_full_run_builds_the_engine_twice_writes_the_files_and_reports_the_tokens_saved(tmp_path, fake_rag, real_litellm):
    code = G.main(["--size", "small", "--quiet", "--ollama-url", fake_rag, "--out-dir", str(tmp_path)])
    folder = Path((tmp_path / "LATEST.txt").read_text().strip())
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and folder.parent == tmp_path and data["steps_count"] == 5
    for arm in ("baseline", "pal"):
        rag = folder / arm / "rag"
        assert sorted(p.name for p in rag.iterdir()) == ["chunker.py", "embedder.py", "prompt.py", "rag.py", "store.py"]
        assert (rag / "rag.py").read_text(encoding="utf-8").startswith("from prompt import build_prompt")
        assert data[arm]["smoke_ran"] and data[arm]["smoke_ok"] and data[arm]["files"] == 5
    b, m = data["baseline"], data["pal"]
    assert b["prompt"] > m["prompt"] and b["main_tokens"] > m["main_tokens"] and data["savings"]["main_tokens"] > 0 and data["savings"]["main_tokens_pct"] > 0
    assert m["helper_tokens"] > 0 and b["helper_tokens"] == 0 and [s["prompt"] for s in data["steps"]["baseline"]] == sorted(s["prompt"] for s in data["steps"]["baseline"])
    assert data["steps"]["pal"][-1]["prompt"] < data["steps"]["baseline"][-1]["prompt"]                 # the plain client's prompt keeps growing; Pal's does not
    assert data["spec_tokens"] > 3500 and data["savings"]["main_tokens_pct"] > 30 and data["hot_turns"] == 6 and data["estimates"]["documents"] > 0
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "**REAL RUN, no simulation.**" in md and "Saved:" in md and "SIMULATED" not in md and "Design-document excerpts instead of the whole document" in md
    chats = [b for b in FakeRag.log if "messages" in b]
    marker = "The design documentation of LocalRAG, for reference:"
    plain_first = [b for b in chats if marker in b["messages"][-1]["content"]]
    assert len(plain_first) == 1 and len(plain_first[0]["messages"][-1]["content"]) > 15000                         # only the plain client pastes the whole document, once
    assert {b["model"] for b in chats} == {"llama3.2:3b"} and all(b["options"]["seed"] == 7 and "num_ctx" in b["options"] for b in chats)       # one model: nothing to swap in memory


def test_without_the_design_document_nothing_is_pasted_and_nothing_is_attached(tmp_path, fake_rag, real_litellm):
    code = G.main(["--size", "small", "--no-spec", "--quiet", "--ollama-url", fake_rag, "--out-dir", str(tmp_path)])
    data = json.loads((Path((tmp_path / "LATEST.txt").read_text().strip()) / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and data["spec_tokens"] == 0 and data["estimates"].get("documents", 0) == 0
    assert not any("design documentation of LocalRAG" in json.dumps(b["messages"]) for b in FakeRag.log if "messages" in b)
    assert data["savings"]["main_tokens_pct"] < 5                                                       # with only a short history to save on, there is little or nothing to save


def test_a_separate_helper_model_is_loaded_and_used_only_by_pal_when_asked_for(tmp_path, fake_rag, real_litellm):
    code = G.main(["--size", "small", "--quiet", "--helper-model", "tiny:1b", "--ollama-url", fake_rag, "--out-dir", str(tmp_path)])
    assert code == 0
    chats = [b for b in FakeRag.log if "messages" in b]
    assert {b["model"] for b in chats} == {"llama3.2:3b", "tiny:1b"} and sum(1 for b in chats if b["messages"][-1]["content"] == "Reply with the single word: ready") == 2
    assert not any(b["model"] == "tiny:1b" and "write the file" in b["messages"][-1]["content"] and "<text>" not in b["messages"][-1]["content"] for b in chats)    # it never writes the code
