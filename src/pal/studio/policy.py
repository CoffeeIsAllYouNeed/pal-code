"""Access policy: what a build may write, and what it may use. Enforced in code, with every refusal explained.

A model's reply is untrusted text. Before any file is written it goes through `Policy.check_write`: the path must stay inside the project folder
(no `..`, absolute paths, drive letters, hidden files, reserved Windows names), the kind of file must be allowed (programs and scripts are not: you add
those yourself), and size and count limits hold. Paths are compared without regard to case, because Windows and macOS treat `Index.html` and
`index.html` as one file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pal.studio.catalog import PERMISSION_BY_ID, tool_needs
from pal.terminal.policy import (  # noqa: E402,F401  (one copy of the path rules)
    ALLOWED_EXTENSIONS,
    ALLOWED_NAMES,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    PolicyError,
    safe_path,
)


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    path: str = ""          # the cleaned path, when there is one


@dataclass
class Policy:
    permissions: dict
    written: dict = field(default_factory=dict)        # lower-cased path -> bytes written
    blocked: list = field(default_factory=list)        # (path, reason)

    def _no(self, path: str, reason: str) -> Decision:
        self.blocked.append((path, reason))
        return Decision(False, reason, path)

    def check_write(self, raw_path: str, content: str) -> Decision:
        try:
            path = safe_path(raw_path)
        except PolicyError as e:
            return self._no(str(raw_path)[:80], str(e))
        if not self.permissions.get("write_project", False):
            return Decision(False, "saving is switched off (dry run): the file is shown but not saved", path)
        size = len(content.encode("utf-8"))
        key = path.lower()
        if size > MAX_FILE_BYTES:
            return self._no(path, f"too big ({size // 1000} KB; the limit is {MAX_FILE_BYTES // 1000} KB)")
        if key in self.written and not self.permissions.get("modify_own_files", False):
            return self._no(path, "it was already written in this build, and changing it is switched off")
        if key not in self.written and len(self.written) >= MAX_FILES:
            return self._no(path, f"too many files (the limit is {MAX_FILES})")
        total = sum(v for k, v in self.written.items() if k != key) + size
        if total > MAX_TOTAL_BYTES:
            return self._no(path, f"the project would be over {MAX_TOTAL_BYTES // 1_000_000} MB")
        self.written[key] = size
        return Decision(True, "", path)

    def check_tool(self, name: str) -> Decision:
        need = tool_needs(name)
        if need and not self.permissions.get(need, False):
            return Decision(False, f"{name} needs '{PERMISSION_BY_ID[need].title}', which is off")
        return Decision(True)

    def allowed_tools(self, names) -> frozenset:
        return frozenset(n for n in names if self.check_tool(n).allowed)
