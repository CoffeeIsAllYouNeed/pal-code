import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localrag.cli.main import build_parser, main, open_engine
from localrag.core.config import Config, config_from_dict
from localrag.core.errors import IndexingError, LocalRAGError, RetrievalError
from localrag.engine import LocalRAG
from localrag.eval import metrics
from localrag.eval.golden import GoldenQuery, load_golden, save_golden
from localrag.eval.report import format_report, to_json
from localrag.eval.runner import evaluate
from localrag.eval.synthetic import distinctive_words, generate_golden

from .conftest import CORPUS

PROJECT = Path(__file__).resolve().parents[1]


# -- the engine --------------------------------------------------------------------------------------------------------------------
def test_ingest_search_and_documents(engine):
    docs = engine.list_documents()
    assert [d["source"] for d in docs] == ["cats.md", "cooking.md", "dogs.md", "util.py"] and all(d["chunks"] >= 1 for d in docs)
    assert engine.stats()["documents"] == 4 and engine.stats()["vector_index"] == "flat" and engine.stats()["trees"] == 4
    doc = engine.get_document("cats.md")
    assert engine.get_document(doc.id) is doc and engine.get_chunk(engine.index.store.chunks_of(doc.id)[0].id).doc_id == doc.id
    with pytest.raises(IndexingError, match="no document 'zzz'"):
        engine.get_document("zzz")


def test_code_is_searchable_by_function_name_and_docstring(engine):
    top = engine.search("function that computes the hypotenuse of a triangle", 2).hits[0].chunk
    assert top.metadata["source"] == "util.py" and top.metadata["symbol"] == "hypotenuse" and "math.sqrt" in top.text
    assert engine.search("Counter class inc", 1).hits[0].chunk.metadata.get("symbol") == "Counter"


def test_reingesting_replaces_a_document_and_deleting_removes_it(engine):
    first = engine.ingest_text("# Cats\n\nCats now only sleep. Nothing else matters to a cat.", "cats.md")
    assert first["doc_id"] != engine.get_document("dogs.md").id and len([d for d in engine.list_documents() if d["source"] == "cats.md"]) == 2
    removed = engine.delete_document(first["doc_id"])
    assert removed == first["chunks"] and engine.stats()["documents"] == 4
    with pytest.raises(IndexingError):
        engine.delete_document(first["doc_id"])
    assert all(h.chunk.doc_id != first["doc_id"] for h in engine.search("cats sleep", 5).hits)


def test_the_tree_finds_sections_and_prints_outlines(engine):
    m = engine.tree_search("feeding adult cats water")
    assert m[0]["source"] == "cats.md" and m[0]["section"] == "Care" and "twice a day" in m[0]["text"]
    assert engine.outline("cats.md").splitlines()[0].startswith("- Cats") and "  - Care" in engine.outline("cats.md")
    with pytest.raises(IndexingError):
        engine.outline("none.md")


def test_save_and_load_restore_an_identical_engine(engine, tmp_path):
    before = [(h.chunk.id, round(h.score, 6)) for h in engine.search("cats purr hunt", 3).hits]
    path = tmp_path / "snap.json"
    size = engine.save(path)
    again = LocalRAG.load(path)
    assert size == path.stat().st_size and [(h.chunk.id, round(h.score, 6)) for h in again.search("cats purr hunt", 3).hits] == before
    assert again.stats()["documents"] == 4 and again.tree_search("training treats")[0]["source"] == "dogs.md" and again.config.to_dict() == engine.config.to_dict()
    assert engine.snapshot_path() == Path(".localrag/snapshot.json") and engine.snapshot_path("x.json") == Path("x.json")


def test_settings_change_behaviour():
    e = LocalRAG(config_from_dict({"chunk": {"chunk_size": 100, "chunk_overlap": 0}, "index": {"vector_index": "ivf", "ivf_lists": 2, "ivf_probes": 2}, "retrieve": {"top_k": 2}}))
    e.ingest_text(" ".join(f"Sentence {i} about subject {i}." for i in range(40)), "long.txt")
    assert e.stats()["chunks"] > 5 and e.stats()["vector_index"] == "ivf" and len(e.search("subject 7").hits) == 2
    with pytest.raises(LocalRAGError):
        LocalRAG(Config(retrieve=config_from_dict({"retrieve": {"top_k": 3}}).retrieve.__class__(top_k=50, candidate_k=10)))


# -- evaluation --------------------------------------------------------------------------------------------------------------------
def test_ranking_metrics_on_a_worked_example():
    rel, ranked = {"a", "c"}, ["b", "a", "d", "c"]
    assert metrics.recall_at_k(rel, ranked, 2) == 0.5 and metrics.recall_at_k(rel, ranked, 4) == 1.0 and metrics.precision_at_k(rel, ranked, 2) == 0.5
    assert metrics.reciprocal_rank(rel, ranked) == 0.5 and metrics.reciprocal_rank(rel, ["x"]) == 0.0
    assert metrics.average_precision(rel, ranked) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert metrics.ndcg_at_k(rel, ranked, 4) == pytest.approx((1 / 1.5849625 + 1 / 2.321928) / (1 + 1 / 1.5849625), rel=1e-3)
    assert metrics.ndcg_at_k({"a"}, ["a"], 1) == 1.0 and metrics.recall_at_k(set(), ranked, 3) == 0.0 and metrics.precision_at_k(rel, [], 3) == 0.0


def test_golden_sets_round_trip_and_report_bad_lines(tmp_path):
    items = [GoldenQuery("alpha beta", {"a.md"}), GoldenQuery("gamma", {"b.md", "c.md"})]
    save_golden(tmp_path / "g.jsonl", items)
    assert load_golden(tmp_path / "g.jsonl") == items
    (tmp_path / "bad.jsonl").write_text('{"query": "ok", "relevant": ["a"]}\n\n{"query": 1}\n')
    with pytest.raises(LocalRAGError, match=r"bad.jsonl:3: not a golden query"):
        load_golden(tmp_path / "bad.jsonl")


def test_synthetic_golden_sets_are_repeatable_and_the_engine_passes_its_own_test(engine):
    a, b = generate_golden(engine, 8, seed=1), generate_golden(engine, 8, seed=1)
    assert a == b and a != generate_golden(engine, 8, seed=2) and all(len(g.query.split()) >= 3 for g in a)
    words = distinctive_words(engine, "Cats purr and hunt mice at night quietly", 3)
    assert len(words) == 3 and "and" not in words
    report = evaluate(engine, generate_golden(engine, 12, seed=0), 5)
    assert report.recall >= 0.9 and report.mrr >= 0.8 and report.failures() == [] and len(report.outcomes) > 0
    text = format_report(report)
    assert "recall@5" in text and "MRR" in text and json.loads(to_json(report))["queries"] == len(report.outcomes)


def test_failures_are_listed_in_the_report(engine):
    report = evaluate(engine, [GoldenQuery("cats purr hunt mice", {"dogs.md"}), GoldenQuery("boil pasta salted water", {"cooking.md"})], 3)
    assert len(report.failures()) == 1 and report.mrr == pytest.approx(0.5) or report.mrr < 1.0
    assert "found nothing relevant" in format_report(report) and "wanted ['dogs.md']" in format_report(report)


# -- the command line --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in CORPUS.items():
        (d / name).write_text(text)
    return d


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_ingest_query_list_stats_tree_and_eval_commands(tmp_path, docs_dir, capsys):
    data = str(tmp_path / "data")
    code, out = run(capsys, "--data-dir", data, "ingest", str(docs_dir))
    assert code == 0 and "cats.md:" in out.out and "util.py:" in out.out and (tmp_path / "data" / "snapshot.json").is_file()
    code, out = run(capsys, "--data-dir", data, "query", "how long to boil pasta", "-k", "1")
    assert code == 0 and out.out.startswith("1. cooking.md")
    code, out = run(capsys, "--data-dir", data, "query", "boil pasta", "--json")
    assert json.loads(out.out)["hits"][0]["chunk_id"] and run(capsys, "--data-dir", data, "query", "qqqq zzzz")[0] == 1
    assert "cats.md" in run(capsys, "--data-dir", data, "list")[1].out
    assert json.loads(run(capsys, "--data-dir", data, "stats")[1].out)["documents"] == 4
    assert "Care" in run(capsys, "--data-dir", data, "tree", "feeding cats")[1].out and run(capsys, "--data-dir", data, "tree", "--document", "dogs.md")[1].out.startswith("- Dogs")
    code, out = run(capsys, "--data-dir", data, "eval", "--queries", "6", "--save-golden", str(tmp_path / "g.jsonl"))
    assert code == 0 and "recall@5" in out.out and len(load_golden(tmp_path / "g.jsonl")) == 6
    assert run(capsys, "--data-dir", data, "eval", "--golden", str(tmp_path / "g.jsonl"), "-k", "3")[0] == 0


def test_errors_are_one_line_messages_with_exit_code_2(tmp_path, capsys):
    code, out = run(capsys, "--data-dir", str(tmp_path / "d"), "ingest", str(tmp_path / "missing"))
    assert code == 2 and out.err.startswith("error: path not found") and "Traceback" not in out.err
    assert run(capsys, "--config", str(tmp_path / "nope.json"), "stats")[0] == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"retrieve": {"top_k": 1}}))
    assert open_engine(str(cfg), str(tmp_path / "d2")).config.retrieve.top_k == 1


def python(*args, stdin=None, cwd=PROJECT):
    return subprocess.run([sys.executable, "-m", "localrag", *args], input=stdin, capture_output=True, text=True, cwd=cwd, env={**os.environ, "PYTHONPATH": str(PROJECT / "src")}, timeout=60)


def test_the_package_runs_as_a_module_and_serves_mcp_over_real_stdio(tmp_path, docs_dir):
    data = str(tmp_path / "data")
    assert python("--data-dir", data, "ingest", str(docs_dir)).returncode == 0 and "localrag" in python("--version").stdout
    rpc = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
           {"jsonrpc": "2.0", "method": "notifications/initialized"},
           {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "how long to boil pasta", "top_k": 1}}}]
    done = python("--data-dir", data, "serve-mcp", stdin="\n".join(json.dumps(m) for m in rpc) + "\n")
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert done.returncode == 0 and [r["id"] for r in replies] == [1, 2] and replies[0]["result"]["serverInfo"]["name"] == "localrag"
    assert json.loads(replies[1]["result"]["content"][0]["text"])["hits"][0]["source"] == "cooking.md" and done.stderr == ""


# -- the documentation must agree with the code ----------------------------------------------------------------------------------------
def test_the_configuration_reference_is_exactly_what_the_code_generates():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_config_doc", PROJECT / "scripts" / "gen_config_doc.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (PROJECT / "docs" / "CONFIG.md").read_text(encoding="utf-8") == gen.render(), "run: python scripts/gen_config_doc.py"


def test_the_mcp_guide_lists_every_tool_and_every_error_code():
    from localrag.mcp import jsonrpc
    from localrag.mcp.factory import build_server
    text = (PROJECT / "docs" / "MCP.md").read_text(encoding="utf-8")
    for tool in build_server(LocalRAG()).tools.names():
        assert f"| `{tool}` |" in text, tool
    for code in (jsonrpc.PARSE_ERROR, jsonrpc.INVALID_REQUEST, jsonrpc.METHOD_NOT_FOUND, jsonrpc.INVALID_PARAMS, jsonrpc.INTERNAL_ERROR,
                 jsonrpc.UNAUTHORIZED, jsonrpc.NOT_INITIALIZED, jsonrpc.RATE_LIMITED, jsonrpc.RESOURCE_NOT_FOUND):
        assert f"| {code} |" in text, code


def test_the_documented_limits_match_the_defaults():
    from localrag.core.config import ServerConfig
    text = (PROJECT / "docs" / "MCP.md").read_text(encoding="utf-8")
    assert f"({ServerConfig().max_message_bytes:,})" in text and f"({ServerConfig().rate_limit_per_minute})" in text
