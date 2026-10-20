import json

import pytest

from localrag.core.config import Config, apply_env, config_from_dict, load_config
from localrag.core.errors import ConfigError, LocalRAGError, ProtocolError
from localrag.core.ids import doc_id_of_chunk, make_chunk_id, make_doc_id, stable_hash
from localrag.core.textutil import (
    char_ngrams, estimate_tokens, normalize_text, sentence_spans, sentence_split, stem_word, tokenize, truncate_to_tokens,
)
from localrag.core.types import Chunk, QueryResult


def test_defaults_are_valid_and_documented_values():
    c = Config().validate()
    assert (c.chunk.chunk_size, c.chunk.chunk_overlap, c.retrieve.top_k, c.index.rrf_k, c.embed.embedding_dim) == (512, 64, 5, 60, 256)


@pytest.mark.parametrize("data,msg", [({"chunk": {"chunk_size": 10}}, "at least 50"), ({"chunk": {"chunk_overlap": 600}}, "less than chunk_size"),
                                      ({"index": {"vector_index": "hnsw"}}, "flat or ivf"), ({"retrieve": {"mmr_lambda": 2.0}}, "between 0 and 1"),
                                      ({"nope": 1}, "unknown setting nope"), ({"chunk": {"chunk_size": "big"}}, "must be int"),
                                      ({"retrieve": {"use_reranker": 1}}, "true or false"), ({"chunk": {"chunk_size": True}}, "must be a number"),
                                      ({"chunk": 3}, "must be an object")])
def test_bad_settings_are_rejected_with_a_clear_message(data, msg):
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(data)


def test_an_int_is_accepted_for_a_float_setting_and_environment_overrides_convert_types():
    assert config_from_dict({"retrieve": {"mmr_lambda": 1}}).retrieve.mmr_lambda == 1.0
    cfg = apply_env(Config(), {"LOCALRAG_CHUNK_SIZE": "300", "LOCALRAG_VECTOR_INDEX": "ivf", "LOCALRAG_TOP_K": "9"})
    assert (cfg.chunk.chunk_size, cfg.index.vector_index, cfg.retrieve.top_k) == (300, "ivf", 9)
    with pytest.raises(ConfigError, match="LOCALRAG_CHUNK_SIZE must be int"):
        apply_env(Config(), {"LOCALRAG_CHUNK_SIZE": "lots"})


def test_load_config_reads_json_and_reports_missing_or_broken_files(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"chunk": {"chunk_size": 200}}))
    assert load_config(p, {}).chunk.chunk_size == 200
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.json")
    p.write_text("{oops")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(p)


def test_config_round_trips_through_a_dictionary():
    cfg = config_from_dict({"retrieve": {"top_k": 8}, "server": {"auth_token": "s3"}})
    assert config_from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_ids_are_stable_and_distinct():
    assert stable_hash("a", "b") == stable_hash("a", "b") != stable_hash("ab")
    d = make_doc_id("x.md", "text")
    assert d.startswith("d_") and make_doc_id("x.md", "text") == d and make_doc_id("y.md", "text") != d
    assert make_chunk_id(d, 3) == f"{d}:0003" and doc_id_of_chunk(make_chunk_id(d, 3)) == d


def test_text_helpers():
    assert normalize_text("a \t b\r\n\r\n\r\n\r\nc\x00") == "a b\n\nc"
    assert tokenize("The cats, and the dogs!") == ["cats", "dogs"] and tokenize("A b cc", drop_stopwords=False, min_len=1) == ["a", "b", "cc"]
    assert [stem_word(w) for w in ("purrs", "purring", "boxes", "parties", "glass", "ran")] == ["pur", "pur", "box", "party", "glass", "ran"]
    assert stem_word("running") == "run" and stem_word("purrs") == stem_word("purring")        # different forms, one stem
    assert tokenize("cats purring", stem=True) == ["cat", "pur"]
    assert char_ngrams("ab", 3) == [" ab", "ab "] and sentence_split("One. Two! Three? four") == ["One.", "Two!", "Three? four"]
    assert sentence_spans("One. Two.") == [(0, 4), (5, 9)] and sentence_spans("") == []
    assert estimate_tokens("") == 0 and estimate_tokens("abcd" * 10) == 10 and estimate_tokens("a") == 1
    assert truncate_to_tokens("word " * 100, 5).endswith(" ...") and truncate_to_tokens("short", 5) == "short"


def test_chunk_and_result_serialise():
    c = Chunk("d:0000", "d", 0, "text", 0, 4, {"source": "a"})
    assert Chunk.from_dict(c.to_dict()) == c
    r = QueryResult("q", [], [], "ctx", 1.234)
    assert r.to_dict()["took_ms"] == 1.23 and r.to_dict()["hits"] == []


def test_every_error_is_a_localrag_error_and_protocol_errors_carry_a_code():
    assert issubclass(ConfigError, LocalRAGError) and ProtocolError("x", -32601).code == -32601 and ProtocolError("x").code == -32600
