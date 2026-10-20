"""What you typed before, per folder: ↑ and ↓ in the input box walk through it, Ctrl+R searches it.

Kept in ~/.pal/projects/<folder id>/history (newest last, at most MAX entries). A line that sets a key (/key ...) is never kept, and
key-shaped strings in any line are hidden first, so the file never holds a secret.
"""
from __future__ import annotations

import json
from pathlib import Path

from pal.paths import pal_home
from pal.terminal.security import redact_secrets
from pal.terminal.workflow import project_id

MAX = 500


class History:
    def __init__(self, folder: Path, home: Path | None = None):
        self.path = (home or pal_home()) / "projects" / project_id(Path(folder)) / "history"
        self.items: list[str] = self._load()

    def _load(self) -> list[str]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, str) and item.strip():
                out.append(item)
        return out[-MAX:]

    def add(self, text: str) -> None:
        text = text.strip()
        if not text or text.split(" ", 1)[0] in ("/key",):
            return
        text, _ = redact_secrets(text)
        if self.items and self.items[-1] == text:
            return
        self.items.append(text)
        self.items = self.items[-MAX:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("".join(json.dumps(x) + "\n" for x in self.items), encoding="utf-8")
        except OSError:
            pass

    def search(self, query: str, before: int) -> int | None:
        """The index of the newest entry older than `before` that contains `query` (case-insensitive), or None."""
        q = query.lower()
        for i in range(min(before, len(self.items)) - 1, -1, -1):
            if q in self.items[i].lower():
                return i
        return None


class Browser:
    """↑/↓ through a History while typing, keeping the unsent draft to come back to."""

    def __init__(self, history: History):
        self.h, self.pos, self.draft = history, len(history.items), ""

    def older(self, current: str) -> str | None:
        if self.pos == len(self.h.items):
            self.draft = current
        if self.pos == 0:
            return None
        self.pos -= 1
        return self.h.items[self.pos]

    def newer(self) -> str | None:
        if self.pos >= len(self.h.items):
            return None
        self.pos += 1
        return self.draft if self.pos == len(self.h.items) else self.h.items[self.pos]

    def find(self, query: str) -> str | None:
        """Ctrl+R: the next older entry containing the query."""
        i = self.h.search(query, self.pos)
        if i is None:
            return None
        self.pos = i
        return self.h.items[i]
