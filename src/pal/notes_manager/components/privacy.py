"""What is never kept as a note: anything that looks like a secret (API keys, tokens, passwords, private keys, long hashes)."""
from __future__ import annotations

import re

_SECRET = re.compile(
    r"sk-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?i:password|passwd|secret|token|api[_ -]?key)\s*[:=]\s*\S{4,}"
    r"|\b[A-Fa-f0-9]{40,}\b|\b[A-Za-z0-9+/]{48,}={0,2}(?=\s|$)")


def looks_secret(text: str) -> bool:
    return bool(_SECRET.search(text))
