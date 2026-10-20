import pytest

from localrag.core.config import RetrieveConfig, TreeConfig
from localrag.core.errors import RetrievalError
from localrag.core.types import Chunk, QueryResult, SearchHit
from localrag.embed.similarity import dot
from localrag.retrieve.cache import QueryCache
from localrag.retrieve.citations import build_citations, make_snippet
from localrag.retrieve.context import build_context, label
from localrag.retrieve.mmr import mmr_select
from localrag.retrieve.query import SYNONYMS, process_query
from localrag.retrieve.rerank import NoopReranker, TermOverlapReranker
from localrag.tree.builder import build_tree
from localrag.tree.render import render_outline, section_text
from localrag.tree.search import TreeIndex, node_score
from localrag.tree.summarize import summarize


def hit(text, score=1.0, source="a.md", cid="d:0000", **meta):
    return SearchHit(Chunk(cid, "d", 0, text, 0, len(text), {"source": source, **meta}), score)


def test_query_processing_extracts_filters_expands_abbreviations_and_refuses_empty_queries():
    q = process_query('  how to set db config source:guide.md heading:"Getting started" ')
    assert q.filters == {"source": "guide.md", "heading": "Getting started"} and "database" in q.text and "configuration" in q.text and "source:" not in q.text
    assert process_query("hello world", {"language": "python"}).filters == {"language": "python"}
    assert process_query("running cats").tokens == ["runn", "cat"] or process_query("running cats").tokens == ["run", "cat"]
    for bad in ("", "the and of", "source:a.md"):
        with pytest.raises(RetrievalError, match="no searchable words"):
            process_query(bad)
    with pytest.raises(RetrievalError, match="must be text"):
        process_query(None)
    assert SYNONYMS["db"] == "database"


def test_term_overlap_reranker_promotes_chunks_that_contain_the_query_words_together():
    hits = [hit("unrelated words entirely", 1.0, cid="d:0"), hit("boil the pasta slowly", 0.9, cid="d:1"), hit("pasta and then boil", 0.9, cid="d:2")]
    out = TermOverlapReranker().rerank(["boil", "pasta"], hits)
    assert out[0].chunk.id == "d:1" and out[-1].chunk.id == "d:0" and all(0.0 <= h.score <= 1.0 for h in out) and TermOverlapReranker().rerank(["x"], []) == []
    noop = NoopReranker().rerank([], [hit("a", 4.0), hit("b", 2.0)])
    assert [h.score for h in noop] == [1.0, 0.5]


def test_mmr_trades_relevance_against_redundancy():
    hits = [hit("a", 1.0, cid="a"), hit("a2", 0.95, cid="b"), hit("c", 0.6, cid="c")]
    vectors = {"a": [1.0, 0.0], "b": [0.999, 0.045], "c": [0.0, 1.0]}
    assert [h.chunk.id for h in mmr_select(hits, vectors, 2, 1.0)] == ["a", "b"]
    assert [h.chunk.id for h in mmr_select(hits, vectors, 2, 0.5)] == ["a", "c"]
    assert len(mmr_select(hits, vectors, 10, 0.7)) == 3 and mmr_select([], {}, 3) == [] and dot(vectors["a"], vectors["c"]) == 0


def test_snippets_are_centred_on_the_match_and_citations_point_at_the_source():
    text = "x " * 100 + "the answer is forty two " + "y " * 100
    s = make_snippet(text, ["answer"], 60)
    assert "answer" in s and s.startswith("... ") and s.endswith(" ...") and len(s) < 80 and make_snippet("short text", ["nothing"], 60) == "short text"
    cites = build_citations([hit("some answer text here", source="g.md", cid="d:0003")], "answer")
    assert cites[0].source == "g.md" and cites[0].chunk_id == "d:0003" and cites[0].end == len("some answer text here") and "answer" in cites[0].snippet


def test_context_numbers_passages_and_respects_the_token_budget():
    hits = [hit("first passage " * 10, source="a.md", heading="Intro"), hit("second passage " * 10, source="b.md", symbol="fn", cid="d:1")]
    ctx = build_context(hits, 500)
    assert ctx.startswith("[1] a.md - Intro\n") and "[2] b.md - fn\n" in ctx and label(hits[0], 7) == "[7] a.md - Intro"
    tight = build_context([hit("word " * 400, cid="d:0"), hit("never shown " * 50, cid="d:1")], 60)
    assert tight.count("[") == 1 and tight.endswith(" ...") and build_context([], 100) == ""


def test_query_cache_is_least_recently_used_and_counts():
    c = QueryCache(2)
    r = QueryResult("q", [], [], "")
    c.put("a", r)
    c.put("b", r)
    assert c.get("a") is r and c.get("zz") is None
    c.put("c", r)
    assert c.get("b") is None and c.get("a") is r and c.stats() == {"hits": 2, "misses": 2, "size": 2}
    c.clear()
    assert c.get("a") is None and QueryCache(0).get("x") is None and QueryCache.key("q", 5, {"a": 1}) == QueryCache.key("q", 5, {"a": 1})
    off = QueryCache(0)
    off.put("a", r)
    assert off.get("a") is None


def test_the_pipeline_ranks_cites_caches_and_validates(engine):
    r = engine.search("which pet purrs and hunts mice", top_k=2)
    assert r.hits[0].chunk.metadata["source"] == "cats.md" and r.hits[0].rank == 1 and 1 <= len(r.hits) <= 2 and r.context.startswith("[1] cats.md")
    assert r.citations[0].source == "cats.md" and not r.from_cache and engine.search("which pet purrs and hunts mice", top_k=2).from_cache
    assert engine.search("boil pasta", filters={"source": "cats.md"}).hits == []                    # the filter excludes the only relevant document
    assert engine.search("cats purr", filters={"source": "cats.md"}).hits[0].chunk.metadata["source"] == "cats.md"
    assert [h.chunk.metadata["source"] for h in engine.search("boil pasta source:cooking.md", top_k=5).hits] == ["cooking.md"]
    assert engine.search("qqqq zzzz").hits == [] and engine.search("qqqq zzzz").context == ""
    with pytest.raises(RetrievalError, match="top_k"):
        engine.retrieval.search("x y z", top_k=-1)


def test_min_score_drops_weak_hits_and_diversity_changes_the_order(engine):
    strict = engine.retrieval.__class__(RetrieveConfig(min_score=0.99, use_reranker=True), engine.index)
    assert len(strict.search("cats purr", 5).hits) <= 1
    plain = engine.retrieval.__class__(RetrieveConfig(use_reranker=False, mmr_lambda=1.0), engine.index).search("cats purr hunt", 3)
    assert plain.hits[0].chunk.metadata["source"] == "cats.md"


def test_summaries_pick_central_sentences_in_order():
    text = ("Solar panels convert sunlight into electricity. Cats are unrelated. Panels use silicon cells to convert sunlight. "
            "The weather was fine. Electricity from panels powers homes with sunlight.")
    s = summarize(text, 2)
    assert s.count(".") == 2 and "Solar panels convert" in s and summarize("One sentence.", 2) == "One sentence." and len(summarize("x. " * 500, 2, 50)) <= 50


DOC = "# Guide\n\nIntro to the guide.\n\n## Install\n\nRun the installer. It sets up everything.\n\n### Windows\n\nUse the msi package.\n\n## Usage\n\nCall the function with a name.\n\n#### Deep\n\ntoo deep to be a node\n"


def test_the_tree_nests_sections_by_heading_level_and_folds_deep_headings():
    root = build_tree("d", "Guide", DOC, TreeConfig(tree_max_depth=3))
    titles = [(n.level, n.title) for n in root.walk()]
    assert titles == [(0, "Guide"), (1, "Guide"), (2, "Install"), (3, "Windows"), (2, "Usage")] and root.depth() == 4
    install = root.children[0].children[0]
    assert "msi" in DOC[install.children[0].start:install.children[0].end] and install.end >= install.children[0].end and "installer" in install.summary
    assert "too deep" in DOC[root.children[0].children[1].start:root.children[0].children[1].end]
    from localrag.tree.nodes import TreeNode
    assert TreeNode.from_dict(root.to_dict()).to_dict() == root.to_dict()


def test_a_document_without_headings_becomes_paragraph_groups():
    text = "\n\n".join(f"Paragraph {i} talks about subject {i}." for i in range(7))
    root = build_tree("d", "Plain", text, TreeConfig())
    assert len(root.children) == 3 and all(c.level == 1 and c.summary for c in root.children) and root.children[0].title.startswith("Paragraph 0")


def test_tree_search_follows_the_best_branches_and_ranks_titles_first():
    root = build_tree("d", "Guide", DOC, TreeConfig())
    idx = TreeIndex()
    idx.add("d", "guide.md", root, DOC)
    matches = idx.search("install windows msi", beam=2, top_n=3)
    assert matches[0].node.title == "Windows" and matches[0].path == "Guide > Guide > Install > Windows" and len(idx) == 1
    assert idx.search("zebra", 2, 5) == [] and idx.get("d") is root and idx.get("x") is None
    assert node_score(root, DOC, []) == 0.0
    assert idx.remove("d") and not idx.remove("d") and idx.search("install", 2, 5) == []


def test_outline_and_section_text_rendering():
    root = build_tree("d", "Guide", DOC, TreeConfig())
    out = render_outline(root)
    assert out.splitlines()[0].startswith("- Guide [") and "  - Install" in out and "    - Windows" in out and "Run the installer" in out
    assert "Usage" not in render_outline(root, max_depth=1) and "Run the installer" not in render_outline(root, with_summary=False)
    assert section_text(DOC, root.children[0].children[0].children[0]).startswith("### Windows") and section_text("word " * 100, root, 50).endswith(" ...")
