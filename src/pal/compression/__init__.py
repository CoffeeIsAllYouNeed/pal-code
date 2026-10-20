"""The Compression component: the NLP prompt compressor and the history compressor (old assistant replies, facts of old turns)."""
from pal.compression.history import (
    HistoryCompressor,
    LayeredHistory,
    compact_reply,
    composition,
    extract_facts,
    facts_block,
)
from pal.compression.pipeline import CompressionPipeline

__all__ = ["CompressionPipeline", "HistoryCompressor", "LayeredHistory", "compact_reply", "composition", "extract_facts", "facts_block"]
