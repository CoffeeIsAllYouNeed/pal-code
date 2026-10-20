"""What every loader shares: the loaded text with its metadata, a base class, and byte decoding that never guesses silently."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from localrag.core.errors import IngestError


@dataclass
class LoadedText:
    """The text a loader extracted and what it learned about the file (title, language, row count, ...)."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def decode_bytes(raw: bytes, name: str = "input") -> str:
    """Decode bytes as UTF-8 (with or without a byte-order mark), then UTF-16, then Latin-1; refuse data that looks binary."""
    if b"\x00" in raw[:4096] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise IngestError(f"{name} looks like a binary file, not text")
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(encoding) if encoding != "utf-16" or raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


class Loader(ABC):
    """Turns the bytes of one kind of file into text. Subclasses list the extensions they handle."""

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, raw: bytes, name: str) -> LoadedText:
        """Return the file's text and metadata, or raise IngestError."""
