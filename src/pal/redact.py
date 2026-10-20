"""Hide API keys in any text that may be shown or logged. Known keys are replaced exactly; strings shaped like common provider
keys are replaced too, because a provider's error message can echo a key we were never told about."""
from __future__ import annotations

import re

_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{10,}|AQ\.[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{12,}"
                     r"|Bearer\s+[A-Za-z0-9._\-]{12,})")


# a key=value pair whose name says it is a secret, as in a query string: ?api_key=abc&q=cats
_PAIR = re.compile(r"(?i)\b(api[_-]?key|apikey|appid|app_id|access[_-]?token|auth[_-]?token|token|secret|password|key)=([^&\s\"']+)")


def redact(text: str, *keys: str | None) -> str:
    for k in keys:
        if k:
            text = text.replace(k, "***")
    return _PAIR.sub(r"\1=***", _SECRET.sub("***", text))
