"""Attachment -> markdown, before anything reaches an API model.

Text-heavy binary/markup formats are converted to markdown because it costs
fewer tokens than the source markup and keeps structure (headings, lists,
tables) that flat extraction loses:

    html/htm/xhtml            pandoc
    docx, odt, rtf            pandoc
    epub                      pandoc
    tex/latex                 pandoc
    pdf (text layer only)     pymupdf4llm if installed, else pypdf
    pptx                      python-pptx
    xlsx                      openpyxl

Anything else is treated as plain text (code, CSV, JSON, .md, ...) and passed
through unchanged. Binary files that are not on the list are rejected.
"""
from __future__ import annotations

from pathlib import PurePath

from pal.input_converters.components import (
    pandoc_converter,
    pdf_converter,
    pptx_converter,
    xlsx_converter,
)
from pal.input_converters.components.base import ConversionError, ConvertedFile

CONVERTIBLE_EXTENSIONS = tuple(sorted(
    set(pandoc_converter.READERS) | {"pdf", "pptx", "xlsx"}
))

# Legacy binary formats we cannot convert; say so instead of "binary not supported".
_LEGACY = {"doc": "docx", "ppt": "pptx", "xls": "xlsx"}


def _ext(name: str) -> str:
    return PurePath(name).suffix.lower().lstrip(".")


def convert_to_markdown(name: str, raw: bytes) -> ConvertedFile:
    """Return the file's content as markdown (or unchanged text). Raises
    ConversionError (a ValueError) with a user-readable message on failure."""
    name = name or "file"
    ext = _ext(name)

    def done(text: str, fmt: str, converted: bool = True) -> ConvertedFile:
        text = text.replace("\r\n", "\n").strip("\n")
        if not text.strip():
            raise ConversionError(f"{name}: no readable text")
        return ConvertedFile(name=name, text=text, source_format=fmt, converted=converted)

    try:
        if ext in pandoc_converter.READERS:
            return done(pandoc_converter.convert(ext, raw), ext)
        if ext == "pdf":
            return done(pdf_converter.convert(raw), "pdf")
        if ext == "pptx":
            return done(pptx_converter.convert(raw), "pptx")
        if ext in ("xlsx", "xlsm"):
            return done(xlsx_converter.convert(raw), "xlsx")
    except ConversionError as e:
        raise ConversionError(f"{name}: {e}") from e

    if ext in _LEGACY:
        raise ConversionError(f"{name}: legacy .{ext} is not supported; save it as .{_LEGACY[ext]} and re-attach")
    if b"\x00" in raw[:4096]:
        raise ConversionError(
            f"{name}: binary files are not supported "
            f"(text, code, CSV, JSON and {', '.join('.' + e for e in CONVERTIBLE_EXTENSIONS)} only)")
    return done(raw.decode("utf-8", errors="replace"), "text", converted=False)
