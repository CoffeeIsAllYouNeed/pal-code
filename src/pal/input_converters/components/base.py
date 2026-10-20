from __future__ import annotations

from dataclasses import dataclass

from pal.errors import UserInputError


class ConversionError(UserInputError):
    """A file could not be turned into markdown. A UserInputError: shown to the user as a per-file error."""


@dataclass
class ConvertedFile:
    name: str
    text: str        # markdown (or the original text for plain-text files)
    source_format: str  # "html", "pdf", "docx", ... or "text"
    converted: bool     # False when the file was already text and passed through


def table_to_markdown(rows: list[list[str]]) -> str:
    """Render rows (first row = header) as a GitHub-style pipe table."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    # drop columns that are empty in every row
    keep = [i for i in range(width) if any(i < len(r) and r[i].strip() for r in rows)]
    if not keep:
        return ""

    def cell(r: list[str], i: int) -> str:
        v = r[i] if i < len(r) else ""
        return v.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()

    lines = ["| " + " | ".join(cell(rows[0], i) for i in keep) + " |",
             "| " + " | ".join("---" for _ in keep) + " |"]
    lines += ["| " + " | ".join(cell(r, i) for i in keep) + " |" for r in rows[1:]]
    return "\n".join(lines)
