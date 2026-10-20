from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pal.errors import UserInputError

# Added to the API system prompt once the session has an indexed project, so the
# API model knows it sees excerpts, not the whole project.
GRAPH_NOTICE = (
    "The user's project is indexed locally as a code graph. It is not pasted in full: each request may end with "
    "<knowledge_graph> holding the graph nodes and code excerpts most relevant to that question. "
    "If they do not cover what you need, name the file or symbol to look up."
)

Runner = Callable[[list, str], tuple]  # (args, cwd) -> (returncode, stdout, stderr)


class GraphError(UserInputError):
    """Indexing or lookup failed. A UserInputError: shown to the user as a per-file error."""


@dataclass
class ProjectInfo:
    name: str
    files: int
    nodes: int
    edges: int
    source_tokens: int                 # rough size of the project source (chars / 4)
    hubs: list = field(default_factory=list)
    top_dirs: list = field(default_factory=list)   # [(dir, file_count)]

    def summary(self) -> str:
        lines = [f"Project '{self.name}' indexed locally as a code graph: {self.files} source files, "
                 f"{self.nodes} nodes, {self.edges} edges (~{self.source_tokens:,} tokens if sent in full)."]
        if self.top_dirs:
            lines.append("Layout: " + ", ".join(f"{d}/ ({n})" for d, n in self.top_dirs))
        if self.hubs:
            lines.append("Main hubs: " + ", ".join(self.hubs))
        lines.append("Its source is not included here. Each request carries the graph excerpts relevant to the question.")
        return "\n".join(lines)


@dataclass
class GraphExcerpts:
    text: str            # the <knowledge_graph> block, ready to append to a request
    nodes: int           # nodes listed in the map
    excerpts: int        # code excerpts included
    tokens: int          # size of `text`
    project_tokens: int  # what sending the whole project would roughly cost
