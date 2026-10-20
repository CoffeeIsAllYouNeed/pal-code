"""Convert a file that is already in a workspace to markdown: the converter as a tool.

Uploads are converted automatically when they arrive. This is the same converter for a file that is already in an agent's workspace
(an upload kept there, a file saved or generated earlier), asked for on demand. The full markdown is saved next to the original
under `converted/` so it can be read in parts, and an existing file is never overwritten.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from pal.input_converters.components.base import ConversionError
from pal.input_converters.pipeline import CONVERTIBLE_EXTENSIONS, convert_to_markdown

OUT_DIR = "converted"


@dataclass
class WorkspaceConversion:
    path: str              # the file that was converted
    text: str              # the whole markdown
    source_format: str
    converted: bool        # False when the file was already text and passed through
    saved: str             # where the markdown was saved ("" if it was not)
    note: str              # why it was not saved, if so


def convertible(path: str) -> bool:
    return PurePosixPath(path).suffix.lower().lstrip(".") in CONVERTIBLE_EXTENSIONS


def convert_workspace_file(path: str, *, read_bytes: Callable[[str], bytes], write_text: Callable[[str, str], object],
                           exists: Callable[[str], bool], max_chars: int = 500_000) -> WorkspaceConversion:
    name = PurePosixPath(path).name
    if not convertible(path):
        raise ConversionError(f"{name}: only {', '.join(CONVERTIBLE_EXTENSIONS)} files can be converted "
                              "(text and code files need no conversion)")
    res = convert_to_markdown(name, read_bytes(path))
    if len(res.text) > max_chars:
        raise ConversionError(f"{name}: the markdown is {len(res.text):,} characters, over the {max_chars:,} limit")
    out = f"{OUT_DIR}/{name}.md"
    if exists(out):
        return WorkspaceConversion(path, res.text, res.source_format, res.converted, "", f"{out} already exists and was not overwritten")
    write_text(out, res.text)
    return WorkspaceConversion(path, res.text, res.source_format, res.converted, out, "")


def describe(r: WorkspaceConversion, preview_chars: int = 2000, indexed: str = "") -> str:
    """The tool's answer: what happened, then the beginning of the markdown."""
    head = f"converted {r.path} ({r.source_format} to markdown): {len(r.text):,} characters."
    head += f" Saved the full text as {r.saved}." if r.saved else f" {r.note}."
    if indexed:
        head += " " + indexed
    cut = r.text[:preview_chars]
    return head + "\n\n" + cut + ("\n[preview ends; read the rest from the saved file]" if len(r.text) > preview_chars else "")
