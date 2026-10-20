"""Stable identifiers: the same content always gets the same id, so re-ingesting a file replaces it instead of duplicating it."""
from __future__ import annotations

import hashlib


def stable_hash(*parts: str, length: int = 12) -> str:
    """A short hex digest of the given strings, the same on every machine and every run."""
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def make_doc_id(source: str, text: str) -> str:
    """The id of a document: its source name and the start of its text decide it."""
    return "d_" + stable_hash(source, text[:2048])


def make_chunk_id(doc_id: str, index: int) -> str:
    """The id of the index-th chunk of a document, for example d_ab12cd34ef56:0003."""
    return f"{doc_id}:{index:04d}"


def doc_id_of_chunk(chunk_id: str) -> str:
    """The document a chunk id belongs to."""
    return chunk_id.rsplit(":", 1)[0]
