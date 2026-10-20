"""Plain text files."""
from __future__ import annotations

from localrag.core.textutil import normalize_text
from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes


class TextLoader(Loader):
    extensions = (".txt", ".text", ".log", ".rst")

    def load(self, raw: bytes, name: str) -> LoadedText:
        text = normalize_text(decode_bytes(raw, name))
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return LoadedText(text, {"format": "text", "title": first[:80]})
