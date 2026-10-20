"""A dependency-free embedder: words, word pairs and character n-grams are hashed into a fixed number of slots (feature hashing)."""
from __future__ import annotations

import math
import zlib
from collections import Counter

from localrag.core.config import EmbedConfig
from localrag.core.textutil import char_ngrams, tokenize
from localrag.embed.base import Embedder
from localrag.embed.idf import IdfTable
from localrag.embed.similarity import normalize


def feature_slot(feature: str, dim: int) -> tuple[int, float]:
    """Where a feature lands in the vector and with which sign; the sign spreads hash collisions in both directions."""
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, 1.0 if (h >> 31) & 1 else -1.0


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, config: EmbedConfig, idf: IdfTable | None = None):
        super().__init__(config.embedding_dim)
        self.config = config
        self.idf = idf

    def _features(self, text: str) -> Counter[str]:
        feats: Counter[str] = Counter()
        words = tokenize(text, stem=True)
        cfg = self.config
        for w in words:
            weight = cfg.word_weight * (self.idf.weight(w) if self.idf else 1.0)
            feats[f"w:{w}"] += weight
        for a, b in zip(words, words[1:], strict=False):
            feats[f"b:{a}_{b}"] += cfg.bigram_weight
        for n in (3, 4):
            for g in char_ngrams(text, n):
                feats[f"c{n}:{g}"] += cfg.char_ngram_weight / 4
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat, weight in self._features(text).items():
                slot, sign = feature_slot(feat, self.dim)
                vec[slot] += sign * (1.0 + math.log(weight)) if weight >= 1.0 else sign * weight
            vectors.append(normalize(vec))
        return vectors
