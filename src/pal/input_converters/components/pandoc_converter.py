"""HTML, DOCX, ODT, RTF, EPUB and LaTeX -> GitHub-flavoured markdown via pandoc."""
from __future__ import annotations

import shutil
import subprocess

from pal.input_converters.components.base import ConversionError

# extension -> pandoc reader
READERS = {
    "html": "html", "htm": "html", "xhtml": "html",
    "docx": "docx", "odt": "odt", "rtf": "rtf",
    "epub": "epub", "tex": "latex", "latex": "latex",
}

_TIMEOUT_S = 60


def available() -> bool:
    return shutil.which("pandoc") is not None


def docx_without_pandoc(raw: bytes) -> str:
    """A Word file as markdown without pandoc: headings, lists and tables from the document's XML (Python's standard library only)."""
    import io
    import xml.etree.ElementTree as ET  # noqa: S405  (expat in Python 3.10+ limits entity expansion)
    import zipfile
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        xml = zipfile.ZipFile(io.BytesIO(raw)).read("word/document.xml")
    except (zipfile.BadZipFile, KeyError):
        raise ConversionError("not a valid Word (.docx) file") from None
    if len(xml) > 50_000_000:
        raise ConversionError("the Word file is too large to read")
    body = ET.fromstring(xml).find(f"{w}body")  # noqa: S314
    if body is None:
        return ""

    def text_of(el) -> str:
        return "".join(t.text or "" for t in el.iter(f"{w}t"))
    out: list[str] = []
    for el in body:
        if el.tag == f"{w}p":
            text = text_of(el).strip()
            if not text:
                continue
            style = el.find(f"{w}pPr/{w}pStyle")
            name = (style.get(f"{w}val") or "") if style is not None else ""
            if name.lower().startswith("heading") and name[-1:].isdigit():
                out.append("#" * min(6, int(name[-1])) + " " + text)
            elif name.lower() == "title":
                out.append("# " + text)
            elif el.find(f"{w}pPr/{w}numPr") is not None or name.lower().startswith("list"):
                if out and out[-1].startswith("- "):
                    out[-1] += "\n- " + text          # one list, not a paragraph per item
                else:
                    out.append("- " + text)
            else:
                out.append(text)
        elif el.tag == f"{w}tbl":
            rows = [[text_of(c).strip().replace("|", "\\|") for c in r.iter(f"{w}tc")] for r in el.iter(f"{w}tr")]
            rows = [r for r in rows if any(r)]
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                out.append("\n".join(["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width] + ["| " + " | ".join(r) + " |" for r in rows[1:]]))
    return "\n\n".join(out)


def convert(ext: str, raw: bytes) -> str:
    reader = READERS[ext]
    if not available() and ext == "docx":
        return docx_without_pandoc(raw)
    if not available():
        raise ConversionError(
            f".{ext} needs pandoc, which is not installed (apt install pandoc / brew install pandoc)")
    cmd = ["pandoc", "-f", reader, "-t", "gfm-raw_html", "--wrap=none"]
    try:
        p = subprocess.run(cmd, input=raw, capture_output=True, timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise ConversionError(f"pandoc timed out after {_TIMEOUT_S}s") from None
    if p.returncode != 0 and ext == "docx":
        return docx_without_pandoc(raw)                  # pandoc rejects some Word files that are still readable
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise ConversionError(f"pandoc could not read the file ({err[-1] if err else 'unknown error'})") from None
    return p.stdout.decode("utf-8", errors="replace")
