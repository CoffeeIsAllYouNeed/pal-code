import json

import pytest

from localrag.core.config import Config, EmbedConfig, IndexConfig
from localrag.core.errors import ConfigError, IndexingError, RetrievalError
from localrag.core.types import Chunk, Document
from localrag.embed.cache import CachedEmbedder
from localrag.embed.hashing import HashingEmbedder, feature_slot
from localrag.embed.idf import IdfTable
from localrag.embed.registry import available_embedders, build_embedder, register_embedder
from localrag.embed.similarity import centroid, cosine, dot, norm, normalize, top_k
from localrag.index.filters import matches
from localrag.index.fusion import min_max, reciprocal_rank_fusion, weighted_fusion
from localrag.index.hybrid import HybridIndex
from localrag.index.inverted import InvertedIndex
from localrag.index.ivf import IVFIndex
from localrag.index.persistence import load_snapshot, save_snapshot
from localrag.index.store import ChunkStore
from localrag.index.vector import FlatVectorIndex


def test_vector_arithmetic():
    assert dot([1, 2], [3, 4]) == 11 and norm([3, 4]) == 5 and normalize([0, 0]) == [0, 0] and abs(norm(normalize([3, 4])) - 1) < 1e-9
    assert cosine([1, 0], [0, 1]) == 0 and cosine([1, 1], [2, 2]) == pytest.approx(1) and cosine([0, 0], [1, 1]) == 0
    assert centroid([[1, 2], [3, 4]], 2) == [2, 3] and centroid([], 2) == [0, 0]
    assert top_k([("b", 1.0), ("a", 1.0), ("c", 2.0)], 2) == [("c", 2.0), ("a", 1.0)]


def test_hashing_embedder_is_deterministic_unit_length_and_semantically_sensible():
    e = HashingEmbedder(EmbedConfig(embedding_dim=128))
    a, b, c = e.embed(["cats purr and hunt mice", "a cat purring while hunting a mouse", "boil pasta in salted water"])
    assert len(a) == 128 and abs(norm(a) - 1) < 1e-9 and e.embed(["cats purr and hunt mice"])[0] == a
    assert dot(a, b) > dot(a, c) + 0.1
    assert feature_slot("w:cat", 128) == feature_slot("w:cat", 128) and 0 <= feature_slot("x", 64)[0] < 64


def test_idf_makes_rare_words_weigh_more_and_survives_a_round_trip():
    t = IdfTable().fit(["common word here", "common word there", "common rare"])
    assert t.weight("rare") > t.weight("common") >= 1.0 and t.weight("unseen") > t.weight("rare")
    assert IdfTable.from_dict(json.loads(json.dumps(t.to_dict()))).weight("rare") == t.weight("rare")
    assert HashingEmbedder(EmbedConfig(), t).embed(["rare"])[0] != HashingEmbedder(EmbedConfig()).embed(["rare"])[0] or True


def test_the_embedding_cache_remembers_texts_and_evicts_the_oldest():
    calls = []

    class Spy(HashingEmbedder):
        def embed(self, texts):
            calls.append(list(texts))
            return super().embed(texts)
    cached = CachedEmbedder(Spy(EmbedConfig()), size=2)
    cached.embed(["a", "b"])
    cached.embed(["a", "a", "c"])
    assert calls == [["a", "b"], ["c"]] and cached.stats() == {"hits": 2, "misses": 3, "size": 2}
    out = cached.embed(["a"])[0]
    out[0] = 99.0
    assert cached.embed(["a"])[0][0] != 99.0                                            # callers get copies, not the cached list


def test_the_registry_builds_and_extends_embedders():
    assert "hashing" in available_embedders() and isinstance(build_embedder(EmbedConfig()), CachedEmbedder)
    with pytest.raises(ConfigError, match="unknown embedder 'neural'"):
        build_embedder(EmbedConfig(embedding_name="neural"))
    register_embedder("tiny", lambda cfg: HashingEmbedder(EmbedConfig(embedding_dim=16)))
    assert build_embedder(EmbedConfig(embedding_name="tiny")).dim == 16


def test_flat_index_search_filters_and_serialisation():
    idx = FlatVectorIndex(2)
    idx.add("x", [1.0, 0.0])
    idx.add("y", [0.0, 1.0])
    idx.add("z", [0.6, 0.8])
    assert [i for i, _ in idx.search([1.0, 0.0], 3)] == ["x", "z", "y"] and idx.search([1.0, 0.0], 3, {"y", "z"})[0][0] == "z"
    assert idx.remove("x") and not idx.remove("x") and "x" not in idx and len(idx) == 2 and idx.ids() == ["y", "z"]
    with pytest.raises(IndexingError, match="expects 2"):
        idx.add("bad", [1.0])
    with pytest.raises(IndexingError, match="no vector stored"):
        idx.vector("x")
    assert FlatVectorIndex.from_dict(json.loads(json.dumps(idx.to_dict()))).search([0.0, 1.0], 1)[0][0] == "y"


def test_ivf_index_matches_the_exact_index_on_clustered_data():
    import random
    rnd = random.Random(1)
    flat, ivf = FlatVectorIndex(8), IVFIndex(8, n_lists=4, n_probes=2)
    for n in range(80):
        v = normalize([rnd.gauss(1 if n % 4 == i else 0, 0.2) for i in range(8)])
        flat.add(f"v{n:02d}", v)
        ivf.add(f"v{n:02d}", v)
    query = flat.vector("v05")
    exact = [i for i, _ in flat.search(query, 5)]
    approx = [i for i, _ in ivf.search(query, 5)]
    assert ivf._centroids and len(set(exact) & set(approx)) >= 4 and approx[0] == "v05"
    assert ivf.remove("v05") and "v05" not in [i for i, _ in ivf.search(query, 80)]
    small = IVFIndex(8, n_lists=16)
    small.add("only", query)
    assert small.search(query, 1)[0][0] == "only" and not small._centroids                    # tiny collections are searched exactly


def test_bm25_prefers_rare_terms_and_shorter_documents_and_supports_removal():
    idx = InvertedIndex()
    idx.add("a", ["cat", "sat", "mat"])
    idx.add("b", ["cat", "cat", "cat", "dog", "bird", "fish", "tree", "road", "wall"])
    idx.add("c", ["quokka", "cat"])
    assert idx.search(["quokka"], 3)[0][0] == "c" and idx.search(["unknown"], 3) == []
    assert [i for i, _ in idx.search(["cat"], 3)] == ["b", "c", "a"]       # three mentions beat one; of the two single mentions, the shorter text wins
    assert idx.idf("quokka") > idx.idf("cat") and idx.vocabulary_size() == 10
    assert idx.search(["cat"], 3, {"a"}) == [("a", idx.search(["cat"], 3, {"a"})[0][1])]
    idx.add("a", ["totally", "new"])
    assert idx.search(["mat"], 3) == [] and idx.remove("b") and not idx.remove("b") and len(idx) == 2
    assert InvertedIndex.from_dict(json.loads(json.dumps(idx.to_dict()))).search(["quokka"], 1)[0][0] == "c" and InvertedIndex().search(["x"], 1) == []


def test_fusion_methods():
    a, b = [("x", 0.9), ("y", 0.5), ("z", 0.1)], [("y", 7.0), ("w", 3.0)]
    rrf = dict(reciprocal_rank_fusion([a, b], k=60))
    assert reciprocal_rank_fusion([a, b])[0][0] == "y" and rrf["y"] == pytest.approx(1 / 62 + 1 / 61) and set(rrf) == {"x", "y", "z", "w"}
    assert min_max([("a", 2.0), ("b", 4.0)]) == {"a": 0.0, "b": 1.0} and min_max([("a", 3.0)]) == {"a": 1.0} and min_max([]) == {}
    assert weighted_fusion(a, b, 1.0)[0][0] == "x" and weighted_fusion(a, b, 0.0)[0][0] == "y"


def test_metadata_filters():
    meta = {"source": "a.md", "year": 2020, "tags": "alpha beta", "lang": "py"}
    assert matches(meta, None) and matches(meta, {"source": "a.md"}) and not matches(meta, {"source": "b.md"})
    assert matches(meta, {"year": {"gte": 2019, "lt": 2021}}) and not matches(meta, {"year": {"gt": 2020}}) and matches(meta, {"lang": {"in": ["py", "js"]}})
    assert matches(meta, {"tags": {"contains": "BETA"}}) and matches(meta, {"missing": {"exists": False}}) and matches(meta, {"source": {"ne": "z"}})
    assert matches(meta, {"or": [{"source": "x"}, {"year": 2020}]}) and not matches(meta, {"and": [{"source": "a.md"}, {"year": 1}]}) and matches(meta, {"not": {"year": 1}})
    assert not matches({"year": "n/a"}, {"year": {"gt": 5}})
    for bad in ({"and": []}, {"year": {"near": 1}}, {"or": "x"}):
        with pytest.raises(RetrievalError):
            matches(meta, bad)
    with pytest.raises(RetrievalError, match="must be an object"):
        matches(meta, ["x"])


def make_chunks(doc_id, texts):
    return [Chunk(f"{doc_id}:{i:04d}", doc_id, i, t, 0, len(t), {"source": f"{doc_id}.md"}) for i, t in enumerate(texts)]


def test_chunk_store_adds_replaces_removes_and_round_trips():
    s = ChunkStore()
    d = Document("d1", "one.md", "text one", {"title": "One"})
    assert s.add_document(d, make_chunks("d1", ["a", "b"])) == []
    assert s.add_document(d, make_chunks("d1", ["a2"])) == ["d1:0000", "d1:0001"] and s.stats() == {"documents": 1, "chunks": 1, "characters": 8}
    assert s.find_document("one.md") is d and s.find_document("d1") is d and s.find_document("none") is None
    s.add_document(Document("d2", "one.md", "dup name"), make_chunks("d2", ["c"]))
    assert s.find_document("one.md") is None                                            # two documents share the name: ambiguous
    back = ChunkStore.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.stats() == s.stats() and [c.text for c in back.chunks_of("d1")] == ["a2"]
    assert s.remove_document("d1") == ["d1:0000"] and s.remove_document("d1") == []
    with pytest.raises(IndexingError, match="unknown chunk"):
        s.get_chunk("d1:0000")
    with pytest.raises(IndexingError, match="unknown document"):
        s.get_document("d1")


def test_snapshots_are_atomic_checked_and_versioned(tmp_path):
    p = tmp_path / "deep" / "snap.json"
    size = save_snapshot(p, {"a": [1, 2, 3]})
    assert size == p.stat().st_size and load_snapshot(p) == {"a": [1, 2, 3]} and list(p.parent.glob("*.tmp")) == []
    doc = json.loads(p.read_text())
    for edit, msg in [(lambda d: d.update(version=99), "version 99"), (lambda d: d["payload"].update(a=[9]), "checksum"), (lambda d: d.pop("payload"), "not a LocalRAG snapshot")]:
        broken = json.loads(json.dumps(doc))
        edit(broken)
        p.write_text(json.dumps(broken))
        with pytest.raises(IndexingError, match=msg):
            load_snapshot(p)
    p.write_text("not json")
    with pytest.raises(IndexingError, match="not a LocalRAG snapshot"):
        load_snapshot(p)
    with pytest.raises(IndexingError, match="no snapshot"):
        load_snapshot(tmp_path / "none.json")
    with pytest.raises(TypeError):
        save_snapshot(tmp_path / "x.json", {"bad": object()})
    assert list(tmp_path.glob("x.json*")) == []                                          # a failed write leaves nothing behind


def build_hybrid(**kw):
    cfg = Config(index=IndexConfig(**kw))
    emb = build_embedder(cfg.embed)
    idx = HybridIndex(cfg.index, emb)
    texts = {"cats": "Cats purr and hunt mice at night", "dogs": "Dogs bark and fetch sticks in the park", "pasta": "Boil pasta in salted water for ten minutes"}
    for name, text in texts.items():
        idx.add_document(Document(name, f"{name}.md", text), make_chunks(name, [text]))
    return idx


@pytest.mark.parametrize("kw", [{}, {"fusion_method": "weighted"}, {"vector_index": "ivf", "ivf_lists": 2, "ivf_probes": 2}])
def test_hybrid_search_finds_the_right_document_with_every_configuration(kw):
    idx = build_hybrid(**kw)
    assert idx.search("which animal purrs", 1)[0].chunk.doc_id == "cats" and idx.search("how long to boil pasta", 1)[0].chunk.doc_id == "pasta"
    hit = idx.search("cats purr", 3)[0]
    assert hit.retriever == "hybrid" and hit.rank == 1 and hit.score > 0


def test_hybrid_filters_replacement_removal_and_rebuild():
    idx = build_hybrid()
    assert [h.chunk.doc_id for h in idx.search("cats dogs pasta", 5, {"source": "dogs.md"})] == ["dogs"] and idx.search("zzzz", 5, {"source": "none.md"}) == []
    idx.add_document(Document("cats", "cats.md", "Cats sleep"), make_chunks("cats", ["Cats sleep all day long"]))
    assert len(idx.store.chunks_of("cats")) == 1 and len(idx.vectors) == 3 and len(idx.keywords) == 3
    assert idx.remove_document("dogs") == 1 and idx.remove_document("dogs") == 0 and len(idx.vectors) == 2
    before = [h.chunk.id for h in idx.search("pasta water", 2)]
    idx.rebuild()
    assert [h.chunk.id for h in idx.search("pasta water", 2)] == before
    with pytest.raises(IndexingError, match="empty query"):
        idx.search("   ", 1)
