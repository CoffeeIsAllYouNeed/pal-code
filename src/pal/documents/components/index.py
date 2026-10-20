"""Per-session document index on top of the SDK: full text goes to the storage manager's files,
chunks go to its vector store. Later turns retrieve only the chunks that match the question."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from pal.config import DocumentsConfig
from pal.documents.components.chunker import Chunk, chunk_markdown
from pal.errors import NotFoundError
from pal.paths import safe_id
from pal.tokens import estimate_tokens

log = logging.getLogger(__name__)


@dataclass
class DocInfo:
    name: str
    slug: str
    tokens: int
    parts: int
    outline: list[str] = field(default_factory=list)
    inline: bool = True   # small enough to send in full on the upload turn


@dataclass
class Hit:
    doc: str
    part: int          # 1-based
    parts: int
    heading: str
    text: str
    score: float | None


def _slug(name: str) -> str:
    return f"{safe_id(name.rsplit('.', 1)[0], default='doc')[:32]}-{hashlib.sha1(name.encode()).hexdigest()[:6]}"  # noqa: S324


class DocumentIndex:
    def __init__(self, sdk, cfg: DocumentsConfig | dict | None = None, count: Callable[[str], int] = estimate_tokens):
        self.sdk = sdk
        self.cfg = DocumentsConfig.coerce(cfg)
        self.count = count
        self.docs: dict[str, DocInfo] = {}

    # -- indexing --------------------------------------------------------------------------
    def add(self, name: str, text: str) -> DocInfo:
        chunks = chunk_markdown(text, self.cfg.chunk_tokens, self.count)
        slug = _slug(name)
        total = sum(c.tokens for c in chunks)
        self.sdk.write(f"docs/{slug}.json", text)
        for c in chunks:
            self.sdk.add_knowledge(f"{slug}:{c.index:04d}", c.text,
                                   {"doc": name, "part": c.index + 1, "parts": len(chunks), "heading": c.heading})
        outline: list[str] = []
        for c in chunks:
            if c.heading and (not outline or not outline[-1].endswith(f" {c.heading}")):
                outline.append(f"{c.index + 1} {c.heading}")
        info = DocInfo(name=name, slug=slug, tokens=total, parts=len(chunks), outline=outline[:25],
                       inline=total <= self.cfg.inline_max_tokens)
        self.docs[name] = info
        log.debug("indexed %s: %d tokens, %d parts", name, total, len(chunks))
        return info

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens for d in self.docs.values())

    def _chunks(self, info: DocInfo) -> list[Chunk]:
        text = self.sdk.read(f"docs/{info.slug}.json")
        return chunk_markdown(text, self.cfg.chunk_tokens, self.count) if isinstance(text, str) else []

    # -- what the API model sees on the upload turn -----------------------------------------
    def upload_view(self, name: str, text: str, tools: bool) -> str:
        """Small documents in full. Large ones as a preview plus an outline."""
        info = self.docs[name]
        if info.inline:
            return text
        chunks = self._chunks(info)
        taken: list[str] = []
        used = 0
        for c in chunks:
            if taken and used + c.tokens > self.cfg.preview_tokens:
                break
            taken.append(c.text)
            used += c.tokens
        rest = info.tokens - used
        how = ("Use search_documents(query) or read_document(name, part) to read more."
               if tools else "The parts relevant to each question are attached to it.")
        outline = ("Outline: " + "; ".join(info.outline) + ". ") if info.outline else ""
        return "\n\n".join(taken) + f"\n\n[Preview only. {rest:,} more tokens in {info.parts} parts. {outline}{how}]"

    def stub(self, name: str) -> str:
        """What the history keeps instead of the file body."""
        i = self.docs[name]
        return f"[file: {name} | {i.tokens:,} tokens | indexed; relevant parts are attached to each question]"

    # -- retrieval ------------------------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None, exclude: set[str] | None = None) -> list[Hit]:
        if not self.docs or not query.strip():
            return []
        k = top_k or self.cfg.retrieve_chunks
        hits = []
        for r in self.sdk.search_knowledge(query, top_k=k * 3):
            score = r.get("score")
            if score is not None and score < self.cfg.min_score:
                continue
            m = r.get("metadata") or {}
            if m.get("doc") not in self.docs or m.get("doc") in (exclude or ()):
                continue
            hits.append(Hit(m["doc"], int(m.get("part", 1)), int(m.get("parts", 1)), m.get("heading", ""),
                            r["text"], score))
        return hits[:k]

    def excerpts(self, query: str, budget: int | None = None, exclude: set[str] | None = None) -> tuple[str, int, int]:
        """(text block, tokens used, number of chunks). Empty when nothing is relevant enough."""
        budget = budget or self.cfg.budget_tokens
        blocks, used = [], 0
        for h in self.search(query, exclude=exclude):
            body = f"[{h.doc} part {h.part}/{h.parts}{': ' + h.heading if h.heading else ''}]\n{h.text}"
            cost = self.count(body)
            if used + cost > budget:
                continue
            blocks.append(body)
            used += cost
        if not blocks:
            return "", 0, 0
        return "<documents>\n" + "\n\n".join(blocks) + "\n</documents>", used, len(blocks)

    def outline_text(self, name: str) -> str:
        """Every part of a document with its heading and size, so a model can decide which part to read."""
        match = next((n for n in self.docs if n.lower() == name.strip().lower()), None)
        if match is None:
            return f"no document named {name!r}; documents: {', '.join(self.docs) or 'none'}"
        info = self.docs[match]
        lines = [f"{info.name}: {info.tokens:,} tokens in {info.parts} parts"]
        for c in self._chunks(info)[:60]:
            lines.append(f"part {c.index + 1} ({c.tokens} tokens): {c.heading or '(no heading)'}")
        if info.parts > 60:
            lines.append(f"[{info.parts - 60} more parts not shown]")
        return "\n".join(lines)

    def read_part(self, name: str, part: int) -> str:
        info = self.docs.get(name) or next((d for n, d in self.docs.items() if n.lower() == name.lower()), None)
        if info is None:
            raise NotFoundError(f"no document named {name!r}. Available: {', '.join(self.docs) or 'none'}")
        chunks = self._chunks(info)
        if not 1 <= part <= len(chunks):
            raise NotFoundError(f"{info.name} has parts 1 to {len(chunks)}")
        c = chunks[part - 1]
        return f"[{info.name} part {part}/{len(chunks)}{': ' + c.heading if c.heading else ''}]\n{c.text}"
