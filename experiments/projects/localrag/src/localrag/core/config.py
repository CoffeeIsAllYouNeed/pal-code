"""Settings. Every default lives here, in one place, and can be overridden by a JSON file or by LOCALRAG_* environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from localrag.core.errors import ConfigError


@dataclass
class ChunkConfig:
    chunk_size: int = 512            # target characters per chunk
    chunk_overlap: int = 64          # characters shared by neighbouring chunks
    chunk_strategy: str = "auto"     # auto, fixed, sentence, markdown or code
    min_chunk_chars: int = 40        # chunks shorter than this are merged into a neighbour


@dataclass
class EmbedConfig:
    embedding_name: str = "hashing"  # which embedder to use
    embedding_dim: int = 256         # length of every vector
    word_weight: float = 1.0         # weight of single words in a vector
    bigram_weight: float = 0.5       # weight of word pairs
    char_ngram_weight: float = 0.3   # weight of character n-grams
    embedding_cache_size: int = 2048  # texts whose vectors are remembered


@dataclass
class IndexConfig:
    vector_index: str = "flat"       # flat (exact) or ivf (approximate)
    ivf_lists: int = 16              # clusters in the ivf index
    ivf_probes: int = 4              # clusters searched per query
    bm25_k1: float = 1.5             # BM25 term-frequency saturation
    bm25_b: float = 0.75             # BM25 length normalisation
    fusion_method: str = "rrf"       # rrf (reciprocal rank fusion) or weighted
    rrf_k: int = 60                  # damping constant of reciprocal rank fusion
    hybrid_alpha: float = 0.5        # weight of the vector scores in weighted fusion


@dataclass
class RetrieveConfig:
    top_k: int = 5                   # hits returned
    candidate_k: int = 30            # candidates fetched before reranking and diversification
    mmr_lambda: float = 0.7          # 1.0 = pure relevance, 0.0 = pure diversity
    min_score: float = 0.15          # hits scoring below this (after reranking, 0 to 1) are dropped
    use_reranker: bool = True        # re-score candidates by how many query terms they contain
    context_tokens: int = 1500       # token budget of the context text handed to a model
    query_cache_size: int = 256      # query results remembered


@dataclass
class TreeConfig:
    tree_max_depth: int = 4          # deepest heading level that becomes a tree node
    summary_sentences: int = 2       # sentences in a node's extractive summary
    tree_beam: int = 2               # branches followed at each level of a tree search


@dataclass
class ServerConfig:
    server_name: str = "localrag"
    server_version: str = "0.1.0"
    protocol_version: str = "2025-06-18"   # newest MCP revision this server speaks
    rate_limit_per_minute: int = 600       # requests allowed per minute
    max_message_bytes: int = 1_000_000     # larger incoming messages are refused
    auth_token: str = ""                   # when set, clients must send it in initialize


@dataclass
class Config:
    data_dir: str = ".localrag"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def validate(self) -> Config:
        """Raise ConfigError for any setting that cannot work; return self so calls can be chained."""
        c = self
        checks = [
            (c.chunk.chunk_size >= 50, "chunk_size must be at least 50"),
            (0 <= c.chunk.chunk_overlap < c.chunk.chunk_size, "chunk_overlap must be at least 0 and less than chunk_size"),
            (c.chunk.chunk_strategy in ("auto", "fixed", "sentence", "markdown", "code"), "chunk_strategy is not known"),
            (c.embed.embedding_dim >= 8, "embedding_dim must be at least 8"),
            (c.index.vector_index in ("flat", "ivf"), "vector_index must be flat or ivf"),
            (c.index.fusion_method in ("rrf", "weighted"), "fusion_method must be rrf or weighted"),
            (0.0 <= c.index.hybrid_alpha <= 1.0, "hybrid_alpha must be between 0 and 1"),
            (c.index.ivf_probes <= c.index.ivf_lists, "ivf_probes cannot exceed ivf_lists"),
            (c.retrieve.top_k >= 1 and c.retrieve.candidate_k >= c.retrieve.top_k, "candidate_k must be at least top_k, and top_k at least 1"),
            (0.0 <= c.retrieve.mmr_lambda <= 1.0, "mmr_lambda must be between 0 and 1"),
            (c.server.rate_limit_per_minute >= 1, "rate_limit_per_minute must be at least 1"),
        ]
        for ok, message in checks:
            if not ok:
                raise ConfigError(message)
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _fill(obj: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown setting {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{key} must be an object")
            _fill(current, value, f"{path}{key}.")
        else:
            if isinstance(current, bool) and not isinstance(value, bool):
                raise ConfigError(f"{path}{key} must be true or false")
            if isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(value, bool):
                raise ConfigError(f"{path}{key} must be a number")
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            if type(value) is not type(current):
                raise ConfigError(f"{path}{key} must be {type(current).__name__}, got {type(value).__name__}")
            setattr(obj, key, value)


def config_from_dict(data: dict[str, Any] | None = None) -> Config:
    """Build a validated Config from nested dictionaries; unknown keys and wrong types are errors."""
    cfg = Config()
    _fill(cfg, data or {})
    return cfg.validate()


_ENV = {"LOCALRAG_DATA_DIR": ("data_dir",), "LOCALRAG_CHUNK_SIZE": ("chunk", "chunk_size"), "LOCALRAG_CHUNK_OVERLAP": ("chunk", "chunk_overlap"),
        "LOCALRAG_EMBEDDING_DIM": ("embed", "embedding_dim"), "LOCALRAG_TOP_K": ("retrieve", "top_k"),
        "LOCALRAG_VECTOR_INDEX": ("index", "vector_index"), "LOCALRAG_AUTH_TOKEN": ("server", "auth_token")}


def apply_env(cfg: Config, environ: dict[str, str] | None = None) -> Config:
    """Override settings from LOCALRAG_* environment variables; values are converted to the setting's type."""
    env = os.environ if environ is None else environ
    for var, path in _ENV.items():
        if var not in env:
            continue
        target = cfg
        for name in path[:-1]:
            target = getattr(target, name)
        current = getattr(target, path[-1])
        try:
            setattr(target, path[-1], type(current)(env[var]))
        except ValueError as e:
            raise ConfigError(f"{var} must be {type(current).__name__}: {env[var]!r}") from e
    return cfg.validate()


def load_config(path: str | Path | None = None, environ: dict[str, str] | None = None) -> Config:
    """Load settings from a JSON file (optional), then apply environment overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"config file is not valid JSON: {e}") from e
    return apply_env(config_from_dict(data), environ)
