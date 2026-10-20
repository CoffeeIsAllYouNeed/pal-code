"""The Pal logo (two people hugging, white and red with a black outline) as pixels for the terminal.

Drawn from the logo image at 24 x 26 pixels with half blocks, two pixel rows per text line, so it shows in any terminal with colour and needs no
image support. W white, R red, K the black outline, . see-through (the terminal's own background).
"""
from __future__ import annotations

from pal.terminal import theme

GRID = [
    '......KKKK.....K........',
    '.....KWWWWWK.KRRRK......',
    '....KWWWWWWWKRRRRRK.....',
    '....KWWWWWWWKRRRRRRK....',
    '....KWWWWWWWWKRRRRRR....',
    '....KWWWWWWWWKRRRRRR....',
    '....KWWWWWWWWKRRRRRR....',
    '....KWWWWWWWWWRRRRRR....',
    '....KWWWWWWWWKRRRRRK....',
    '...KRKWWWWWWWKRRRRKKK...',
    '..KRRKKWWWWWWKRRRKWWK...',
    '..KRRRKKWWWWKRRRRKWWK...',
    '...KRRRRKKKKRRRRRKKKK...',
    '..KKKRRRRRRRRRRRRRRRRK..',
    '..WWWKRRRRRRRRKKWKRRRR..',
    '.KWWWWKRRRRRRKWWWKRRRRK.',
    '.KWWWWWKKRRRKWWWWKRRRRR.',
    'KWWWWWWWWKKKWWWWKRRRRRRK',
    'KWWWWWWWWWWWWWWWKRRRRRRK',
    'KWWWWWWWWWWWWWWKRRRRRRRK',
    'KWWWWWWWWWWWWWKRRRRRRRRK',
    'KWWWWWWWWWWWWKRRRRRRRRRK',
    'KWWWWWWWWWKKRRRRRRRRRRRK',
    'KWWWWWWWWKRRRRRRRRRRRRRK',
    'KWWWWWWWWKRRRRRRRRRRRRRK',
    'KKKKKKKKKKKKKKKKKKKKKKKK',
]
COLOURS = {"W": theme.WHITE, "R": theme.RED, "K": "#000000"}


def logo_lines(grid: list[str] | None = None) -> list[list[tuple[str, str]]]:
    """Lines of (characters, style). The style is 'fg bg' in hex, or '' for nothing drawn: rich and prompt_toolkit both read it after a small change."""
    g = grid or GRID
    out: list[list[tuple[str, str]]] = []
    for y in range(0, len(g), 2):
        top, bottom = g[y], g[y + 1] if y + 1 < len(g) else "." * len(g[y])
        line: list[tuple[str, str]] = []
        for t, b in zip(top, bottom):
            ct, cb = COLOURS.get(t), COLOURS.get(b)
            if ct is None and cb is None:
                line.append((" ", ""))
            elif ct is None:
                line.append(("▄", f"{cb}|"))
            elif cb is None or ct == cb:
                line.append(("▀" if cb is None else "█", f"{ct}|"))
            else:
                line.append(("▀", f"{ct}|{cb}"))
        out.append(line)
    return out


def rich_style(style: str) -> str:
    fg, _, bg = style.partition("|")
    return (fg + (f" on {bg}" if bg else "")) if fg else ""


def ptk_style(style: str) -> str:
    fg, _, bg = style.partition("|")
    return (f"fg:{fg}" + (f" bg:{bg}" if bg else "")) if fg else ""


# The start screen's logo: drawn from the logo image as solid pixels (half blocks, two pixel rows per text line), at two sizes.
LOGO = ['.....KKKK...........', '....KWWWWK.KRRR.....', '....WWWWWWKRRRRR....', '...KWWWWWWWKRRRRK...', '...KWWWWWWWKRRRRK...',
        '...KWWWWWWWKRRRRK...', '....WWWWWWWKRRRRK...', '...KKWWWWWWKRRRRK...', '..KRKWWWWWWKRRRKW...', '..KRRKWWWWKRRRKWWK..',
        '..KRRRKKKKRRRRKKK...', '..KKRRRRRRRRRKRRRK..', '.KWWKRRRRRRRKWKRRRK.', '.WWWWWKRRRRKWWKRRRR.', '.WWWWWWKRKKWWWKRRRRK',
        'KWWWWWWWWWWWWKRRRRRK', 'KWWWWWWWWWWWKRRRRRRK', 'KWWWWWWWWWWKRRRRRRRK', 'KWWWWWWWWWKRRRRRRRRK', 'KWWWWWWWKRRRRRRRRRRK',
        'KWWWWWWWKRRRRRRRRRRK']
LOGO_SMALL = ['....KK..K.....', '...WWWWKRRR...', '...WWWWKRRRK..', '..KWWWWWRRRR..', '..KWWWWWRRRR..', '..KWWWWWRRRK..', '..RKWWWKRRKW..',
              '..RRKWKRRRKK..', '.KKRRRRRRKRRK.', '.WWWRRRRWWRRR.', '.WWWWKRWWWRRRR', 'WWWWWWWWWKRRRR', 'WWWWWWWWKRRRRR', 'WWWWWWWKRRRRRR',
              'WWWWWWRRRRRRRR', 'KWWWWKRRRRRRRK']

# The wordmark "pal code" in pixel letters. Rows 0-1 ascenders, 2-6 the letters, 7-8 the tail of the p.
FONT = {
    "p": [".....", ".....", "####.", "#...#", "#...#", "#...#", "####.", "#....", "#....", "....."],
    "a": [".....", ".....", ".###.", "....#", ".####", "#...#", ".####", ".....", ".....", "....."],
    "l": ["##.", ".#.", ".#.", ".#.", ".#.", ".#.", ".##", "...", "...", "..."],
    "c": [".....", ".....", ".####", "#....", "#....", "#....", ".####", ".....", ".....", "....."],
    "o": [".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.", ".....", ".....", "....."],
    "d": ["....#", "....#", ".####", "#...#", "#...#", "#...#", ".####", ".....", ".....", "....."],
    "e": [".....", ".....", ".###.", "#...#", "#####", "#....", ".####", ".....", ".....", "....."],
    "i": ["#", ".", "#", "#", "#", "#", "#", ".", ".", "."],
    "w": [".......", ".......", "#.....#", "#.....#", "#..#..#", "#.#.#.#", ".#...#.", ".......", ".......", "......."],
    "b": ["#....", "#....", "####.", "#...#", "#...#", "#...#", "####.", ".....", ".....", "....."],
    "n": [".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", ".....", "....."],
    "h": ["#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", ".....", "....."],
}
WORDS = {"cli": "pal code", "web": "pal code", "bench": "pal code"}    # the wordmark everywhere: "pal" in white, "code" in red


def word_for(kind: str = "cli") -> list[str]:
    text = WORDS.get(kind, WORDS["cli"])
    return word_grid(text, "WWW " + "R" * (len(text) - 4))


def word_grid(text: str = "pal code", colours: str = "WWW RRRR") -> list[str]:
    rows = [""] * 10
    for ch, col in zip(text, colours):
        if ch == " ":
            rows = [r + "..." for r in rows]
            continue
        rows = [r + FONT[ch][y].replace("#", col) + "." for y, r in enumerate(rows)]
    width = max(len(r.rstrip(".")) for r in rows)
    return [r[:width] for r in rows]


def _see_through(grid: list[str]) -> list[str]:
    """The black outline is left to the terminal's own background, so it matches any dark theme."""
    return [r.replace("K", ".") for r in grid]


def banner(size: str = "full", kind: str = "cli") -> list[list[tuple[str, str]]]:
    """The start screen's mark as lines of (characters, 'fg|bg') runs: the logo with the wordmark beside it ('full' or 'small'), or the
    wordmark alone ('word')."""
    word = logo_lines(word_for(kind))
    if size == "word":
        return word
    art = logo_lines(_see_through(LOGO if size == "full" else LOGO_SMALL))
    gap, top = 4, (len(art) - len(word)) // 2
    blank_art = [(" " * len(art[0]), "")]
    out = []
    for i in range(max(len(art), len(word) + top)):
        left = art[i] if i < len(art) else blank_art
        right = word[i - top] if 0 <= i - top < len(word) else []
        line = list(left) + [(" " * gap, "")] + list(right)
        width = sum(len(c) for c, _ in line)
        out.append(line + [(" " * (len(art[0]) + gap + len(word[0]) - width), "")])     # every line the same width, so centring keeps the shape
    return out


BANNER_ROWS = {size: len(banner(size)) for size in ("full", "small", "word")}
