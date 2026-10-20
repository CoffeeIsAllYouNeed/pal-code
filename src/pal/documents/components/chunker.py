"""Split markdown into chunks of about `target_tokens`, at heading and paragraph boundaries.
Pure and deterministic: the same text always gives the same chunks, so part numbers are stable."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pal.tokens import estimate_tokens

_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    index: int        # 0-based
    heading: str      # nearest heading above the chunk ("" if none)
    text: str
    tokens: int


def _split_long(paragraph: str, target: int, count: Callable[[str], int]) -> list[str]:
    """Break one oversized paragraph on line, then sentence, then character boundaries."""
    if count(paragraph) <= target:
        return [paragraph]
    pieces: list[str] = []
    for line in paragraph.split("\n"):
        pieces.extend(re.split(r"(?<=[.!?])\s+", line) if count(line) > target else [line])
    out: list[str] = []
    cur = ""
    for piece in pieces:
        while count(piece) > target:  # a single huge "sentence": hard cut
            cut = target * 4
            out.append(piece[:cut])
            piece = piece[cut:]
        if cur and count(cur + "\n" + piece) > target:
            out.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def chunk_markdown(text: str, target_tokens: int = 350, count: Callable[[str], int] = estimate_tokens) -> list[Chunk]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []
    buf_heading = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            body = "\n\n".join(buf).strip()
            if body:
                chunks.append(Chunk(len(chunks), buf_heading, body, count(body)))
        buf = []

    for para in re.split(r"\n\s*\n", text):
        para = para.strip("\n")
        if not para.strip():
            continue
        m = _HEADING.match(para.split("\n", 1)[0])
        if m:
            flush()  # a heading starts a new chunk
            heading = m.group(1)
        for piece in _split_long(para, target_tokens, count):
            heading_only = len(buf) == 1 and _HEADING.match(buf[0]) is not None
            if buf and not heading_only and count("\n\n".join(buf + [piece])) > target_tokens:
                flush()
            if not buf:
                buf_heading = heading
            buf.append(piece)
    flush()
    return chunks
