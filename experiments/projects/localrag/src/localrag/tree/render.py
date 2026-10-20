"""Showing a tree: an indented outline for people, and the text of one section for a model."""
from __future__ import annotations

from localrag.tree.nodes import TreeNode


def render_outline(root: TreeNode, max_depth: int = 4, with_summary: bool = True) -> str:
    """The tree as an indented list of titles, each followed by its summary on the next line when requested."""
    lines: list[str] = []

    def visit(node: TreeNode, depth: int) -> None:
        if depth > max_depth:
            return
        if node.level > 0:
            lines.append("  " * (node.level - 1) + f"- {node.title} [{node.start}:{node.end}]")
            if with_summary and node.summary:
                lines.append("  " * (node.level - 1) + f"  {node.summary}")
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 0)
    return "\n".join(lines)


def section_text(text: str, node: TreeNode, max_chars: int = 2000) -> str:
    """The document text of one section, cut to max_chars."""
    body = text[node.start:node.end].strip()
    return body if len(body) <= max_chars else body[:max_chars].rsplit(" ", 1)[0] + " ..."
