"""Chooses the loader for a file name and exposes one function to turn bytes into text."""
from __future__ import annotations

from pathlib import PurePosixPath

from localrag.core.errors import IngestError
from localrag.ingest.loaders.base import LoadedText, Loader
from localrag.ingest.loaders.code import CodeLoader
from localrag.ingest.loaders.csv_ import CsvLoader
from localrag.ingest.loaders.html import HtmlLoader
from localrag.ingest.loaders.json_ import JsonLoader
from localrag.ingest.loaders.markdown import MarkdownLoader
from localrag.ingest.loaders.text import TextLoader

_LOADERS: list[Loader] = [MarkdownLoader(), HtmlLoader(), CsvLoader(), JsonLoader(), CodeLoader(), TextLoader()]
BY_EXTENSION: dict[str, Loader] = {ext: loader for loader in _LOADERS for ext in loader.extensions}


def supported_extensions() -> list[str]:
    """Every file extension LocalRAG can ingest, sorted."""
    return sorted(BY_EXTENSION)


def get_loader(name: str) -> Loader:
    """The loader for a file name, chosen by its extension; unknown extensions are an IngestError."""
    ext = PurePosixPath(name).suffix.lower()
    if ext not in BY_EXTENSION:
        raise IngestError(f"cannot ingest {name}: {ext or 'no extension'} is not supported (supported: {', '.join(supported_extensions())})")
    return BY_EXTENSION[ext]


def load_bytes(raw: bytes, name: str) -> LoadedText:
    """Extract the text of a file given its bytes and its name."""
    loaded = get_loader(name).load(raw, name)
    if not loaded.text.strip():
        raise IngestError(f"{name} has no text to index")
    return loaded
