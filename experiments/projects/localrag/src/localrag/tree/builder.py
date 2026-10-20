"""Builds a document tree from markdown-style headings; a document without headings becomes a root with one child per few paragraphs."""
from __future__ import annotations

import re

from localrag.core.config import TreeConfig
from localrag.tree.nodes import TreeNode
from localrag.tree.summarize import summarize

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


def build_tree(doc_id: str, title: str, text: str, config: TreeConfig) -> TreeNode:
    """The tree of a document. Headings deeper than tree_max_depth are folded into their parent's text."""
    marks = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text) if len(m.group(1)) <= config.tree_max_depth]
    root = TreeNode(f"{doc_id}#0", title, 0, 0, len(text))
    if not marks:
        return _paragraph_tree(doc_id, root, text, config)
    stack, counter = [root], 0
    for i, (pos, level, heading) in enumerate(marks):
        counter += 1
        end = next((m[0] for m in marks[i + 1:] if m[1] <= level), len(text))
        node = TreeNode(f"{doc_id}#{counter}", heading, level, pos, end)
        while stack[-1].level >= level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    for node in root.walk():
        body_end = node.children[0].start if node.children else node.end
        own = text[node.start:body_end]
        node.summary = summarize(_HEADING.sub("", own, count=1) if node.level else own, config.summary_sentences)
    return root


def _paragraph_tree(doc_id: str, root: TreeNode, text: str, config: TreeConfig, group: int = 3) -> TreeNode:
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)]
    for n, i in enumerate(range(0, len(spans), group), start=1):
        part = spans[i:i + group]
        start, end = part[0][0], part[-1][1]
        first = " ".join(text[start:end].split())[:60]
        root.children.append(TreeNode(f"{doc_id}#{n}", first, 1, start, end, summarize(text[start:end], config.summary_sentences)))
    root.summary = summarize(text, config.summary_sentences)
    return root
