import pytest

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.documents import DocumentIndex, chunk_markdown
from pal.errors import NotFoundError
from pal.sdk.pipeline import AgentSDK
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.tool_manager.pipeline import ToolManagerPipeline

BIG = "\n\n".join(f"## Part {i}\n\n" + f"Facts about topic{i} and subject{i}. " * 40 for i in range(8))


@pytest.fixture
def sdk(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "s"))
    return AgentSDK("chat-x", None, sm, ToolManagerPipeline(), AccessManagerPipeline())


# -- chunker -------------------------------------------------------------------------------------
def test_chunks_follow_headings_and_stay_near_the_target():
    chunks = chunk_markdown(BIG, 150)
    assert len(chunks) >= 8 and [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.tokens <= 150 + 10 for c in chunks) and {c.heading for c in chunks} >= {"Part 0", "Part 7"}


def test_chunker_is_deterministic_and_lossless_enough():
    a, b = chunk_markdown(BIG, 150), chunk_markdown(BIG, 150)
    assert a == b
    assert "topic5" in " ".join(c.text for c in a)


def test_chunker_edge_cases():
    assert chunk_markdown("") == [] and chunk_markdown("  \n\n ") == []
    assert len(chunk_markdown("one line")) == 1
    assert [c.text for c in chunk_markdown("# T\r\n\r\nbody\r\n")] == ["# T\n\nbody"]
    huge_word = "x" * 5000
    assert all(c.tokens <= 110 for c in chunk_markdown(huge_word, 100))
    assert len(chunk_markdown("a. " * 2000, 100)) > 5


def test_heading_only_chunks_are_not_produced():
    for c in chunk_markdown(BIG, 100):
        assert c.text.strip().count("\n") > 0 or not c.text.startswith("#")


# -- index -----------------------------------------------------------------------------------------
def test_small_documents_are_inline_and_large_ones_are_previewed(sdk):
    idx = DocumentIndex(sdk, {"inline_max_tokens": 300, "preview_tokens": 120, "chunk_tokens": 100})
    idx.add("small.md", "tiny text")
    idx.add("big.md", BIG)
    assert idx.docs["small.md"].inline and not idx.docs["big.md"].inline
    assert idx.upload_view("small.md", "tiny text", True) == "tiny text"
    view = idx.upload_view("big.md", BIG, True)
    assert "Preview only" in view and "Outline:" in view and "search_documents" in view and len(view) < len(BIG) / 3
    assert "attached to it" in idx.upload_view("big.md", BIG, False)


def test_search_returns_only_relevant_chunks_within_budget(sdk):
    idx = DocumentIndex(sdk, {"chunk_tokens": 100, "budget_tokens": 300, "retrieve_chunks": 2})
    idx.add("big.md", BIG)
    text, used, n = idx.excerpts("tell me about topic6 and subject6")
    assert n >= 1 and used <= 300 and "topic6" in text and "topic1 " not in text and text.startswith("<documents>")
    assert idx.excerpts("zebra quantum") == ("", 0, 0)
    assert idx.excerpts("   ") == ("", 0, 0)


def test_excluded_documents_are_not_retrieved(sdk):
    idx = DocumentIndex(sdk, {"chunk_tokens": 100})
    idx.add("a.md", "alpha unique1 words here")
    idx.add("b.md", "alpha unique2 words here")
    names = {h.doc for h in idx.search("alpha words", exclude={"a.md"})}
    assert names == {"b.md"}


def test_read_part_is_exact_and_validates(sdk):
    idx = DocumentIndex(sdk, {"chunk_tokens": 100})
    info = idx.add("Big.md", BIG)
    assert "topic0" in idx.read_part("Big.md", 1) and "topic7" in idx.read_part("big.md", info.parts)  # name is case-insensitive
    assert idx.read_part("Big.md", 1).startswith("[Big.md part 1/")
    with pytest.raises(NotFoundError, match="parts 1 to"):
        idx.read_part("Big.md", info.parts + 1)
    with pytest.raises(NotFoundError, match="no document named"):
        idx.read_part("other.md", 1)


def test_full_text_lives_in_the_storage_manager_and_same_names_do_not_collide(sdk):
    idx = DocumentIndex(sdk)
    a, b = idx.add("a b.md", "one"), idx.add("a_b.md", "two")
    assert a.slug != b.slug and sdk.read(f"docs/{a.slug}.json") == "one"
    assert idx.total_tokens == a.tokens + b.tokens


def test_hostile_file_names_are_safe(sdk):
    idx = DocumentIndex(sdk)
    for name in ("../../evil.md", "C:\\x\\y.md", "con.md", "a\x00b.md", "é.md"):
        info = idx.add(name, "body text")
        assert sdk.read(f"docs/{info.slug}.json") == "body text"
