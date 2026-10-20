"""Colours and the wordmark. Black, red and white."""
from __future__ import annotations

RED = "#E50914"
RED_HI = "#FF4D57"
WHITE = "#FFFFFF"
INK = "#E8E8E8"
MUTED = "#8A8A8A"
PANEL = "#1A1A1A"
BG = "#000000"

_GLYPHS = {                       # chunky pixel letters, 10 rows: ascender rows 0-1, x-height rows 2-7, descender rows 8-9
    "p": ["      ", "      ", "XXXXX ", "XXXXXX", "XX  XX", "XX  XX", "XXXXXX", "XXXXX ", "XX    ", "XX    "],
    "a": ["      ", "      ", " XXXX ", "    XX", " XXXXX", "XX  XX", "XX  XX", " XXXXX", "      ", "      "],
    "l": ["XXX ", " XX ", " XX ", " XX ", " XX ", " XX ", " XX ", " XXX", "    ", "    "],
    ".": ["  ", "  ", "  ", "  ", "  ", "  ", "XX", "XX", "  ", "  "],
}


def wordmark(text: str = "pal", dot: bool = True) -> list[list[tuple[str, str]]]:
    """The wordmark as lines of (characters, colour): pixels drawn with half blocks (two pixel rows per line), two cells per pixel. The final dot is red."""
    cols: list[tuple[list[str], str]] = [(_GLYPHS[ch], WHITE) for ch in text]
    if dot:
        cols.append((_GLYPHS["."], RED))
    lines: list[list[tuple[str, str]]] = []
    for r in range(0, 10, 2):
        line: list[tuple[str, str]] = []
        for i, (glyph, colour) in enumerate(cols):
            top, bottom = glyph[r], glyph[r + 1]
            chars = ""
            for x in range(max(len(top), len(bottom))):
                t = x < len(top) and top[x] == "X"
                b = x < len(bottom) and bottom[x] == "X"
                chars += ("\u2588" if t and b else "\u2580" if t else "\u2584" if b else " ") * 2      # two cells per pixel: a cell is about twice as tall as wide
            line.append((chars + ("  " if i < len(cols) - 1 else ""), colour))
        lines.append(line)
    return lines


# The loader: three dots, the bright one moving along ("···", "●··", "·●·", "··●"), in Pal's red. Registered with rich as "pal".
SPINNER = "pal"
try:
    from rich.spinner import SPINNERS as _SPINNERS
    _SPINNERS[SPINNER] = {"interval": 140, "frames": ["\u00B7\u00B7\u00B7", "\u25CF\u00B7\u00B7", "\u00B7\u25CF\u00B7", "\u00B7\u00B7\u25CF"]}
except Exception:  # noqa: BLE001  (an older rich: its own dots)
    SPINNER = "dots"
