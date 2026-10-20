"""Text-layer PDFs -> markdown. Uses pymupdf4llm (headings, tables) when it is
installed, otherwise pypdf with one section per page. Scans have no text layer
and are rejected: there is no OCR here."""
from __future__ import annotations

import io

from pal.input_converters.components.base import ConversionError


def _with_pymupdf(raw: bytes) -> str | None:
    try:
        import pymupdf  # noqa: F401
        import pymupdf4llm
    except ImportError:
        return None
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


def _with_pypdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ConversionError("PDF support needs pypdf (pip install pypdf)") from None
    pages = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(raw)).pages]
    if len(pages) == 1:
        return pages[0]
    return "\n\n".join(f"## Page {i}\n\n{t}" for i, t in enumerate(pages, 1) if t)


def convert(raw: bytes) -> str:
    try:
        text = _with_pymupdf(raw)
        if text is None or not text.strip():
            text = _with_pypdf(raw)
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(f"could not read PDF ({e})") from e
    if not text.strip():
        raise ConversionError("PDF has no text layer (scanned?); OCR is not supported")
    return text
