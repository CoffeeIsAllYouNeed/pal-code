"""Golden sets: queries paired with the sources that should answer them, stored as one JSON object per line."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from localrag.core.errors import LocalRAGError


@dataclass
class GoldenQuery:
    query: str
    relevant: set[str] = field(default_factory=set)       # source names that contain the answer


def save_golden(path: str | Path, items: list[GoldenQuery]) -> None:
    """Write a golden set, one query per line."""
    Path(path).write_text("\n".join(json.dumps({"query": g.query, "relevant": sorted(g.relevant)}) for g in items) + "\n", encoding="utf-8")


def load_golden(path: str | Path) -> list[GoldenQuery]:
    """Read a golden set written by save_golden; a malformed line is an error that names the line."""
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            items.append(GoldenQuery(str(data["query"]), set(data["relevant"])))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise LocalRAGError(f"{path}:{number}: not a golden query ({e})") from e
    return items
