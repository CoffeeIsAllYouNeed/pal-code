import json

import pytest

from localrag.core.config import ChunkConfig
from localrag.core.errors import IngestError
from localrag.core.types import Document
from localrag.ingest.chunkers.code import CodeChunker
from localrag.ingest.chunkers.fixed import FixedChunker
from localrag.ingest.chunkers.markdown import MarkdownChunker
from localrag.ingest.chunkers.registry import choose_chunker, strategy_for
from localrag.ingest.chunkers.sentence import SentenceChunker
from localrag.ingest.dedupe import NearDuplicateFilter, jaccard, shingles
from localrag.ingest.loaders.base import decode_bytes
from localrag.ingest.loaders.registry import get_loader, load_bytes, supported_extensions
from localrag.ingest.pipeline import IngestPipeline

LONG = " ".join(f"Sentence number {i} says something distinct about topic {i}." for i in range(60))


def doc(text, source="x.txt"):
    return Document("d_test", source, text)


def test_loaders_read_each_format():
    assert load_bytes(b"# Title\n\nbody", "a.md").metadata["title"] == "Title"
    assert load_bytes(b"---\nauthor: Ann\n---\n# T\n\nbody", "a.md").metadata["author"] == "Ann"
    html = load_bytes(b"<html><title>Page</title><script>bad()</script><h2>Sub</h2><p>Hello <b>world</b></p><li>item</li></html>", "a.html")
    assert html.metadata["title"] == "Page" and "## Sub" in html.text and "Hello world" in html.text and "bad()" not in html.text and "- item" in html.text
    assert load_bytes(b"name,age\nAnn,30\nBob,\n", "a.csv").text == "name: Ann; age: 30\nname: Bob"
    assert load_bytes(b'{"a": {"b": [1, 2]}, "c": "x"}', "a.json").text.splitlines() == ["a.b[0]: 1", "a.b[1]: 2", "c: x"]
    assert load_bytes(b'{"a": 1}\n{"a": 2}', "a.jsonl").text.splitlines() == ["record0.a: 1", "record1.a: 2"]
    code = load_bytes(b"x = 1\r\n", "a.py")
    assert code.metadata["language"] == "python" and code.text == "x = 1"
    assert load_bytes(b"first line\nsecond", "a.txt").metadata["title"] == "first line"


def test_loader_errors_name_the_file_and_the_problem():
    for raw, name, msg in [(b"x", "a.xyz", "not supported"), (b"\x00\x01binary\x00", "a.txt", "binary"), (b"   ", "a.txt", "no text"),
                           (b"{bad", "a.json", "not valid JSON"), (b"a,b\n", "a.csv", "no text"), (b"", "a.csv", "no header")]:
        with pytest.raises(IngestError, match=msg):
            load_bytes(raw, name)
    assert ".md" in supported_extensions() and get_loader("X.MD").extensions


def test_decoding_handles_bom_utf16_and_latin1():
    assert decode_bytes("héllo".encode("utf-8-sig")) == "héllo"
    assert decode_bytes("héllo".encode("utf-16")) == "héllo"
    assert decode_bytes("héllo".encode("latin-1")) == "héllo"


def test_fixed_chunker_covers_the_text_with_overlap_and_makes_progress():
    cfg = ChunkConfig(chunk_size=100, chunk_overlap=20)
    spans = FixedChunker(cfg).split(LONG)
    assert spans[0].start == 0 and spans[-1].end == len(LONG) and all(s.end - s.start <= 100 for s in spans)
    assert all(b.start < a.end for a, b in zip(spans, spans[1:], strict=False)) and all(b.start > a.start for a, b in zip(spans, spans[1:], strict=False))
    assert len(FixedChunker(ChunkConfig(chunk_size=100, chunk_overlap=0)).split("x" * 250)) == 3


def test_sentence_chunker_keeps_sentences_whole_and_handles_one_huge_sentence():
    cfg = ChunkConfig(chunk_size=200, chunk_overlap=0)
    for s in SentenceChunker(cfg).split(LONG):
        assert LONG[s.start:s.end].strip().endswith(".")
    huge = "word " * 400
    assert len(SentenceChunker(ChunkConfig(chunk_size=100)).split(huge)) > 5


def test_markdown_chunker_never_mixes_sections_and_records_the_heading_path():
    text = "intro\n\n# A\n\nalpha text\n\n## B\n\nbeta text\n\n# C\n\ngamma text"
    chunks = MarkdownChunker(ChunkConfig(chunk_size=500, min_chunk_chars=5)).chunk(doc(text, "m.md"))
    heads = [c.metadata.get("heading") for c in chunks]
    assert heads == [None, "A", "A > B", "C"] and "beta" in chunks[2].text and "gamma" not in chunks[2].text
    big = "# Big\n\n" + LONG
    assert {c.metadata["heading"] for c in MarkdownChunker(ChunkConfig(chunk_size=200)).chunk(doc(big, "b.md"))} == {"Big"}


def test_code_chunker_splits_python_by_definition_and_falls_back_for_broken_code():
    src = "import os\n\nX = 1\n\n\ndef first():\n    return 1\n\n\n@deco\nclass Second:\n    def m(self):\n        return 2\n"
    spans = CodeChunker(ChunkConfig(chunk_size=500, min_chunk_chars=1)).split(src)
    assert [s.meta["symbol"] for s in spans] == ["module header", "first", "Second"] and "@deco" in src[spans[2].start:spans[2].end]
    assert len(CodeChunker(ChunkConfig(chunk_size=100, min_chunk_chars=1)).split("def f(:\n" + "x = 1\n\n" * 60)) > 1
    long_fn = "def big():\n" + "    y = 1\n" * 200
    parts = CodeChunker(ChunkConfig(chunk_size=200, min_chunk_chars=1)).split(long_fn)
    assert len(parts) > 2 and all(p.meta["symbol"] == "big" for p in parts)


def test_small_chunks_are_merged_and_offsets_point_back_into_the_text():
    text = "A long enough opening paragraph that stands alone here.\n\nTiny.\n\nAnother long enough paragraph that also stands alone."
    chunks = FixedChunker(ChunkConfig(chunk_size=60, chunk_overlap=0, min_chunk_chars=20)).chunk(doc(text))
    assert all(len(c.text) >= 20 or c is chunks[0] for c in chunks) and all(text[c.start:c.end] == c.text for c in chunks)


def test_strategy_selection():
    assert [strategy_for(n) for n in ("a.md", "a.html", "a.py", "a.txt", "a.csv")] == ["markdown", "markdown", "code", "sentence", "sentence"]
    assert choose_chunker(ChunkConfig(), "a.py").name == "code" and choose_chunker(ChunkConfig(chunk_strategy="fixed"), "a.py").name == "fixed"
    with pytest.raises(IngestError, match="unknown chunking strategy"):
        choose_chunker(ChunkConfig(chunk_strategy="bogus"), "a.txt")


def test_near_duplicates_are_detected():
    a = "the quick brown fox jumps over the lazy dog every single morning before breakfast"
    assert jaccard(shingles(a), shingles(a)) == 1.0 and jaccard(shingles(a), shingles("completely different words here")) == 0.0
    f = NearDuplicateFilter(0.8)
    assert f.check_and_add(a) and not f.check_and_add(a + "!") and f.check_and_add("another sentence with entirely other content words") and len(f) == 2
    assert jaccard(frozenset(), frozenset({"x"})) == 0.0 and len(shingles("two words")) == 1


def test_pipeline_ingests_files_and_drops_repeated_chunks():
    p = IngestPipeline(ChunkConfig(chunk_size=120, chunk_overlap=0))
    repeated = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. " * 2 + "\n\n") * 4
    r = p.ingest_text(repeated + "A final different paragraph about something else entirely, long enough to count.", "r.md")
    assert r.duplicates_dropped >= 1 and [c.index for c in r.chunks] == list(range(len(r.chunks))) and all(c.id.endswith(f":{c.index:04d}") for c in r.chunks)
    with pytest.raises(IngestError, match="larger than"):
        p.ingest_bytes(b"x" * 20_000_001, "big.txt")


def test_ingest_path_reads_folders_skips_hidden_and_unsupported_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nsome text in a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("text in b")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.md").write_text("# no")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    (tmp_path / "bad.json").write_text("{oops")
    results = IngestPipeline(ChunkConfig()).ingest_path(tmp_path)
    assert sorted(r.document.source for r in results) == ["a.md", "sub/b.txt"]
    assert len(IngestPipeline(ChunkConfig()).ingest_path(tmp_path, recursive=False)) == 1
    with pytest.raises(IngestError, match="path not found"):
        IngestPipeline(ChunkConfig()).ingest_path(tmp_path / "nope")
    only_unsupported = tmp_path / "only"
    only_unsupported.mkdir()
    (only_unsupported / "pic.png").write_bytes(b"\x89PNG")
    with pytest.raises(IngestError, match="no supported files"):
        IngestPipeline(ChunkConfig()).ingest_path(only_unsupported)
    single = tmp_path / "bad.json"
    with pytest.raises(IngestError, match="not valid JSON"):
        IngestPipeline(ChunkConfig()).ingest_path(single)
    assert json.loads(json.dumps(results[0].metadata))["chunker"]


def test_source_names_use_forward_slashes_on_every_platform(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.md").write_text("# Deep\n\nsome text")
    sources = [r.document.source for r in IngestPipeline(ChunkConfig()).ingest_path(tmp_path)]
    assert sources == ["a/b/deep.md"] and not any("\\" in s for s in sources)
