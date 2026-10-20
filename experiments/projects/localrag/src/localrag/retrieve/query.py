"""Query processing: tidy the text, pull out inline filters such as source:guide.md, and add a few common synonyms."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from localrag.core.errors import RetrievalError
from localrag.core.textutil import normalize_text, tokenize

FILTER_KEYS = ("source", "language", "format", "heading", "symbol")
SYNONYMS = {"db": "database", "auth": "authentication", "config": "configuration", "doc": "document", "docs": "documentation",
            "repo": "repository", "env": "environment", "func": "function", "param": "parameter", "args": "arguments"}
_INLINE = re.compile(r"\b(" + "|".join(FILTER_KEYS) + r"):(\"[^\"]+\"|\S+)")


@dataclass
class ProcessedQuery:
    """A query ready for the indexes: the text to search for, its stemmed tokens, and any filters found inside it."""
    original: str
    text: str
    tokens: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


def process_query(raw: str, extra_filters: dict[str, Any] | None = None) -> ProcessedQuery:
    """Normalise a query, move key:value filters out of the text, expand abbreviations, and refuse an empty result."""
    if not isinstance(raw, str):
        raise RetrievalError("a query must be text")
    text = normalize_text(raw)
    filters: dict[str, Any] = dict(extra_filters or {})
    for key, value in _INLINE.findall(text):
        filters[key] = value.strip('"')
    text = _INLINE.sub(" ", text)
    words = text.split()
    extra = [SYNONYMS[w.lower()] for w in words if w.lower() in SYNONYMS]
    text = " ".join(words + extra).strip()
    tokens = tokenize(text, stem=True)
    if not tokens:
        raise RetrievalError("the query has no searchable words")
    return ProcessedQuery(raw, text, tokens, filters)
