"""Reading a model's reply: which files did it write? Tolerant of the many ways models label a code block."""
from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)\n?```", re.S)
_NAME = re.compile(r"(?<![\w.-])((?:[A-Za-z]:)?[/\\]?(?:(?:\.\.|[\w-]+)[/\\])*[\w-][\w.-]*\.[A-Za-z0-9]{1,5})(?![\w/\\-])")
_INFO = re.compile(r"(?:title|filename|file|path)\s*=\s*[\"']?([^\s\"']+)", re.I)
_FIRST_LINE = re.compile(r"^\s*(?:<!--|//|#|/\*)\s*(?:file(?:name)?|path)\s*:\s*(\S+?)\s*(?:-->|\*/)?\s*$", re.I)


@dataclass
class ParsedFile:
    path: str
    content: str
    how: str = ""


def _hint(text_before: str) -> str:
    """A file name in the last non-empty line before a code block, or ''."""
    for line in reversed(text_before.splitlines()):
        if line.strip():
            names = _NAME.findall(line)
            return names[-1] if names else ""
    return ""


def parse_files(reply: str, default_name: str = "") -> tuple[list[ParsedFile], int]:
    """(the files in the reply, how many code blocks had no usable name). The name comes from the line before the block ("### FILE: index.html",
    "**style.css**", "`app.py`"), from the block's info string (title="x"), or from a first-line comment (<!-- file: x -->). A reply with exactly one
    unnamed block uses `default_name`, if there is one."""
    found: list[ParsedFile] = []
    unnamed = 0
    blocks = list(_FENCE.finditer(reply))
    for i, m in enumerate(blocks):
        before = reply[(blocks[i - 1].end() if i else 0):m.start()]
        body = m.group("body")
        name, how = "", ""
        info = _INFO.search(m.group("info"))
        if info:
            name, how = info.group(1), "info string"
        if not name:
            first = _FIRST_LINE.match(body.split("\n", 1)[0])
            if first:
                name, how, body = first.group(1), "first-line comment", body.split("\n", 1)[1] if "\n" in body else ""
        if not name:
            name, how = _hint(before), "line before"
        if not name and len(blocks) == 1 and default_name:
            name, how = default_name, "default"
        if not name:
            unnamed += 1
            continue
        found.append(ParsedFile(name.strip("`*"), body.rstrip("\n") + "\n", how))
    return found, unnamed
