"""Which paths and files Pal may write. Enforced in code, with every refusal explained.

A model's reply is untrusted text. Before any file is written its path goes through `safe_path`: it must stay inside the folder
(no `..`, absolute paths, drive letters, hidden files, reserved Windows names), the kind of file must be allowed (programs and scripts are not: you add
those yourself). Paths are compared without regard to case, because Windows and macOS treat `Index.html` and
`index.html` as one file.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

ALLOWED_EXTENSIONS = frozenset({".html", ".htm", ".css", ".js", ".mjs", ".json", ".md", ".txt", ".py", ".csv", ".svg", ".yml", ".yaml", ".toml", ".ini", ".cfg",
                                ".xml", ".ts", ".tsx", ".jsx", ".sql"})
ALLOWED_NAMES = frozenset({".gitignore"})
MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES = 200_000, 60, 2_000_000
_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})
_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_. \-]*$")
_DRIVE = re.compile(r"^[A-Za-z]:")


class PolicyError(ValueError):
    """A path or file the policy refuses; the message says why, in plain words."""


def safe_path(raw: str, extensions: frozenset = ALLOWED_EXTENSIONS, names: frozenset = ALLOWED_NAMES) -> str:
    """The path as a clean relative POSIX path inside the project folder, or PolicyError. `extensions` and `names` say which kinds of file are allowed."""
    if not isinstance(raw, str) or not raw.strip():
        raise PolicyError("the file has no name")
    text = raw.strip().replace("\\", "/")
    if "\x00" in text or any(ord(c) < 32 for c in text):
        raise PolicyError("the file name has control characters")
    if text.startswith("/") or _DRIVE.match(text):
        raise PolicyError("absolute paths are not allowed: files go inside the project folder")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts:
        raise PolicyError("the file has no name")
    if any(p == ".." for p in parts):
        raise PolicyError("'..' is not allowed: files must stay inside the project folder")
    if len(text) > 120 or any(len(p) > 60 for p in parts):
        raise PolicyError("the file name is too long")
    for p in parts:
        if p.rstrip(". ") != p:
            raise PolicyError("a name may not end with a dot or a space")
        if p.split(".")[0].lower() in _RESERVED:
            raise PolicyError(f"'{p}' is a reserved name on Windows")
        if p.startswith(".") and p not in names:
            raise PolicyError("hidden files are not allowed")
        if p not in names and not _PART.match(p):
            raise PolicyError(f"'{p}' has characters that are not allowed")
    name = parts[-1]
    if name not in names and PurePosixPath(name).suffix.lower() not in extensions:
        raise PolicyError(f"'{name}': this kind of file is not allowed (programs and scripts are added by you, not by the bot)")
    return "/".join(parts)
