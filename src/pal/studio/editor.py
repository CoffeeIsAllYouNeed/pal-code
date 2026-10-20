"""The code editor behind the Studio: open a built project, change a file by hand, or ask Pal to change it.

Every "ask" goes through the same token-saving client the builds use (PalClient), kept alive for the whole editing session, so the history is paged
and compressed and the project's other files are indexed once instead of pasted again. Nothing Pal writes is saved until the person accepts it:
`ask` only proposes (old text, new text, a diff); `save_file` is the one place a file changes, and it applies the same path and size rules as a build.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pal.errors import UserInputError
from pal.studio.builder import FORMAT
from pal.studio.policy import MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES, PolicyError, safe_path
from pal.terminal.files import _FENCE, parse_files
from pal.terminal.session import diff_lines  # noqa: F401  (one diff, shared with the terminal)
from pal.tokens import estimate_tokens

SYSTEM = "You are a careful senior developer editing an existing project. You change only what was asked, and keep every file consistent with the others."
MAX_DIFF_LINES = 400
MAX_INSTRUCTION = 4000


def lookup(files: dict[str, str], path: str) -> str | None:
    """The stored name of `path`, compared without regard to case (Windows and macOS treat Index.html and index.html as one file)."""
    low = path.lower()
    return next((p for p in files if p.lower() == low), None)


def save_file(state, path: str, content: str) -> dict:
    """Write one file into a finished build (memory, and the project folder when the build was saved). Raises UserInputError with a plain reason."""
    if state.build is None:
        raise UserInputError("this build has no files yet")
    try:
        clean = safe_path(path)
    except PolicyError as e:
        raise UserInputError(str(e)) from None
    if not isinstance(content, str):
        raise UserInputError("the file's text must be a string")
    size = len(content.encode("utf-8"))
    if size > MAX_FILE_BYTES:
        raise UserInputError(f"too big ({size // 1000} KB; the limit is {MAX_FILE_BYTES // 1000} KB)")
    files = state.build.files
    name = lookup(files, clean) or clean
    if name not in files and len(files) >= MAX_FILES:
        raise UserInputError(f"too many files (the limit is {MAX_FILES})")
    if sum(len(c.encode("utf-8")) for p, c in files.items() if p != name) + size > MAX_TOTAL_BYTES:
        raise UserInputError(f"the project would be over {MAX_TOTAL_BYTES // 1_000_000} MB")
    if state.workspace is not None:
        target = (state.workspace / name).resolve()
        if state.workspace.resolve() not in target.parents:
            raise UserInputError("files must stay inside the project folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
    files[name] = content
    return {"path": name, "bytes": size, "saved_to_disk": state.workspace is not None}


@dataclass
class EditorSession:
    """One editing conversation about one build. Holds the token-saving client and the running totals."""
    client: Any
    product: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    sent: dict[str, str] = field(default_factory=dict)       # path -> the text Pal's index last received
    history_tokens: int = 0                                  # what a plain chat would carry along: every earlier request and reply
    asks: int = 0
    pal_tokens: int = 0
    helper_tokens: int = 0
    plain_tokens: int = 0

    def ask(self, files: dict[str, str], path: str, instruction: str) -> dict:
        if not self.lock.acquire(blocking=False):
            raise UserInputError("Pal is still working on your last request: wait for it to finish")
        try:
            return self._ask(files, path, instruction)
        finally:
            self.lock.release()

    def _ask(self, files: dict[str, str], path: str, instruction: str) -> dict:
        name = lookup(files, path) if path else None
        others = {p: c for p, c in files.items() if p != name}
        changed = [{"name": p, "text": c, "kind": "document"} for p, c in others.items() if self.sent.get(p) != c]
        listing = "\n".join(f"- {p} ({len(c.encode('utf-8')):,} bytes)" for p, c in sorted(files.items()))
        parts = [f"Project: {self.product}. Files:\n{listing}", "The other files are available to you as reference documents; read them only if the change needs them."]
        if name:
            parts.append(f"The file open in the editor is {name}. Its current content:\n```\n{files[name]}\n```")
        parts += [f"Change requested: {instruction}",
                  "Reply with the COMPLETE new content of every file you change (never a fragment, never '...'), and leave other files out. " + FORMAT
                  + " If the request is only a question, answer in a few sentences and write no files."]
        message = "\n\n".join(parts)
        # what a plain chat client would have sent for this same request: the system line, the whole project, every earlier turn, and the request
        plain_prompt = estimate_tokens(SYSTEM) + sum(estimate_tokens(c) for c in files.values()) + estimate_tokens(listing) + self.history_tokens + estimate_tokens(instruction)
        reply = self.client.ask(message, changed or None)
        for p, c in others.items():
            self.sent[p] = c
        self.asks += 1
        self.pal_tokens += reply.prompt_tokens + reply.completion_tokens
        self.helper_tokens += reply.helper_tokens
        self.plain_tokens += plain_prompt + reply.completion_tokens
        self.history_tokens += estimate_tokens(instruction) + estimate_tokens(reply.text)

        parsed, _unnamed = parse_files(reply.text, default_name=name or "")
        proposals, refused = [], []
        for pf in parsed:
            try:
                clean = safe_path(pf.path)
            except PolicyError as e:
                refused.append({"path": pf.path[:80], "reason": str(e)})
                continue
            if len(pf.content.encode("utf-8")) > MAX_FILE_BYTES:
                refused.append({"path": clean, "reason": f"too big (the limit is {MAX_FILE_BYTES // 1000} KB)"})
                continue
            existing = lookup(files, clean)
            old = files[existing] if existing else ""
            if existing and old == pf.content:
                continue
            proposals.append({"path": existing or clean, "new": not existing, "old": old, "content": pf.content, "diff": diff_lines(old, pf.content)})
        note = " ".join(m.group(1).strip() for m in re.finditer(r"^NOTE:\s*(.+)$", reply.text, re.M))
        talk = re.sub(r"^#+\s*FILE:.*$|^NOTE:.*$", "", _FENCE.sub("", reply.text), flags=re.M | re.I).strip()
        return {"files": proposals, "refused": refused, "note": note, "message": talk[:1500],
                "usage": {"prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens, "helper_tokens": reply.helper_tokens, "seconds": round(reply.seconds, 1),
                          "plain_estimate": plain_prompt + reply.completion_tokens},
                "session": {"asks": self.asks, "pal_tokens": self.pal_tokens, "helper_tokens": self.helper_tokens, "plain_estimate": self.plain_tokens,
                            "saved_pct": round(100.0 * (self.plain_tokens - self.pal_tokens) / self.plain_tokens, 1) if self.plain_tokens else 0.0}}

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


def new_session_dir(root: Path, run_id: str, arm: str) -> Path:
    return root / run_id / f"{arm}-edit-home"
