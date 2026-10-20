"""A document tree: sections nested by heading level, each with a short summary. Used for retrieval that needs no vectors."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """One section: its title, heading level, character span in the document, an extractive summary and its sub-sections."""
    id: str
    title: str
    level: int
    start: int
    end: int
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)

    def walk(self) -> Iterator[TreeNode]:
        """This node and every node below it, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def depth(self) -> int:
        """The number of levels in the tree below and including this node."""
        return 1 + max((c.depth() for c in self.children), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "level": self.level, "start": self.start, "end": self.end, "summary": self.summary,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        return cls(data["id"], data["title"], int(data["level"]), int(data["start"]), int(data["end"]), data.get("summary", ""),
                   [cls.from_dict(c) for c in data.get("children", [])])
