"""Markdown files, including a simple key: value front matter block."""
from __future__ import annotations

import re

from localrag.core.textutil import normalize_text
from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_H1 = re.compile(r"^#\s+(.+)$", re.M)


class MarkdownLoader(Loader):
    extensions = (".md", ".markdown")

    def load(self, raw: bytes, name: str) -> LoadedText:
        text = decode_bytes(raw, name).replace("\r\n", "\n")
        meta: dict[str, object] = {"format": "markdown"}
        m = _FRONT.match(text)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().lower()] = value.strip().strip("\"'")
            text = text[m.end():]
        text = normalize_text(text)
        h1 = _H1.search(text)
        meta.setdefault("title", h1.group(1).strip() if h1 else name)
        return LoadedText(text, meta)
