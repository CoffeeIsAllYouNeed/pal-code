"""Tree search: no vectors. Score each section by how well its title, summary and opening words match the query, and follow the best branches."""
from __future__ import annotations

from dataclasses import dataclass

from localrag.core.textutil import tokenize
from localrag.tree.nodes import TreeNode


@dataclass
class TreeMatch:
    """A section that matched: which document, the section, its score and the path of titles that leads to it."""
    doc_id: str
    source: str
    node: TreeNode
    score: float
    path: str


def node_score(node: TreeNode, text: str, query_tokens: list[str]) -> float:
    """Title matches count most, then the summary, then the first 300 characters of the section's own text."""
    wanted = set(query_tokens)
    if not wanted:
        return 0.0

    def overlap(s: str) -> float:
        return len(wanted & set(tokenize(s, stem=True))) / len(wanted)
    return 2.0 * overlap(node.title) + 1.0 * overlap(node.summary) + 0.5 * overlap(text[node.start:node.start + 300])


class TreeIndex:
    def __init__(self) -> None:
        self._trees: dict[str, tuple[str, TreeNode, str]] = {}

    def add(self, doc_id: str, source: str, root: TreeNode, text: str) -> None:
        self._trees[doc_id] = (source, root, text)

    def remove(self, doc_id: str) -> bool:
        return self._trees.pop(doc_id, None) is not None

    def get(self, doc_id: str) -> TreeNode | None:
        found = self._trees.get(doc_id)
        return found[1] if found else None

    def search(self, query: str, beam: int = 2, top_n: int = 5) -> list[TreeMatch]:
        """Beam search each tree: at every level keep the `beam` children whose subtree matches best, and collect every visited section."""
        tokens = tokenize(query, stem=True)
        matches: list[TreeMatch] = []
        for doc_id, (source, root, text) in self._trees.items():
            frontier = [(root, root.title)]
            while frontier:
                nxt: list[tuple[float, TreeNode, str]] = []
                for node, path in frontier:
                    for child in node.children:
                        cpath = f"{path} > {child.title}"
                        own = node_score(child, text, tokens)
                        below = max((node_score(d, text, tokens) for d in child.walk()), default=0.0)
                        if own > 0:
                            matches.append(TreeMatch(doc_id, source, child, own, cpath))
                        nxt.append((max(own, below), child, cpath))
                nxt.sort(key=lambda t: (-t[0], t[1].start))
                frontier = [(n, p) for score, n, p in nxt[:beam] if score > 0]
        matches.sort(key=lambda m: (-m.score, m.doc_id, m.node.start))
        return matches[:top_n]

    def __len__(self) -> int:
        return len(self._trees)
