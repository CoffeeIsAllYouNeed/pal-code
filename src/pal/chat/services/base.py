"""What every third-party service tool shares: how it is described, how it reaches the network, and how its errors read.

A service tool is a function the model may ask for by name. Rules that hold for all of them:
  * off unless named in `tools.services`, and offered only when the message looks like it needs the tool and its key is set;
  * the key comes from the environment, is sent in a header where the service allows it, and is never in the tool's arguments,
    its result, an error message or a log (the OS kernel interface logs a URL without its query string);
  * every request goes through the OS kernel interface, so private addresses, redirects and sizes are checked there;
  * the result is plain text for the model, and the tool prompt already says tool results are data, never instructions.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pal.errors import PalError
from pal.redact import redact
from pal.tool_manager.components.base import ToolParam

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
MEDIA_DIRS = ("uploads/", "generated/")
_RAPID_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*\.p\.rapidapi\.com$")


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    signature: str                       # as the model is told: name(arg, arg)
    params: tuple                        # ToolParam, ...
    run: Callable[..., str]
    summary: str                         # one line, for the docs and `pal doctor`
    cue: re.Pattern | None               # does the message look like it needs this tool?
    key_env: str | None = None           # environment variable holding the key; None = no key needed
    needs: str = ""                      # "image" or "audio": only offered when the session workspace holds such an upload
    sends: str = ""                      # who receives what, in one phrase


def param(name: str, typ: type, default: Any = ..., ) -> ToolParam:
    return ToolParam(name, typ) if default is ... else ToolParam(name, typ, required=False, default=default)


def clamp(n: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(n)))
    except (TypeError, ValueError):
        return lo


def clean_text(value: Any, what: str, maxlen: int = 300) -> str:
    s = re.sub(r"[\x00-\x1f]+", " ", str(value)).strip()
    s = " ".join(s.split())
    if not s:
        raise PalError(f"give {what}")
    return s[:maxlen]


def explain(service: str, status: int, text: str, key_env: str | None = None) -> str:
    """What an HTTP failure means, in words, without ever echoing a key."""
    body = redact(" ".join(text.split()))[:200]
    if status in (401, 403):
        if "not subscribed" in text.lower():
            return f"{service}: your key is not subscribed to this RapidAPI listing. Subscribe to it on RapidAPI, or set tools.rapidapi_hosts for a listing you have"
        return f"{service} rejected the key{f' in {key_env}' if key_env else ''}; check that it is valid and allowed to use this API"
    if status == 402:
        return f"{service}: payment is required or the quota is used up"
    if status == 404:
        return f"{service}: nothing found, or the API address is no longer valid (HTTP 404)"
    if status == 410:
        return f"{service}: this API has been retired (HTTP 410)"
    if status == 429:
        return f"rate limit or quota reached for {service}; wait and try again later"
    if status >= 500:
        return f"{service} is having problems (HTTP {status}); try again later"
    return f"{service} refused the request (HTTP {status}): {body}"


class ServiceCtx:
    """What a service tool may use. Built fresh for each call."""

    def __init__(self, env: Mapping[str, str], cfg: Any, os_call: Callable[[str, dict], Any]):
        self.env, self.cfg, self._os = env, cfg, os_call
        self.saved: list[str] = []            # workspace files this call made, so the turn can offer them for download

    def key(self, var: str) -> str:
        value = (self.env.get(var) or "").strip()
        if not value:
            raise PalError(f"{var} is not set, so this service cannot be used")
        return value

    def os(self, op: str, params: dict) -> Any:
        return self._os(op, params)

    def net(self, url: str, **params: Any) -> dict:
        return self._os("net_request", {"url": url, **params})

    def fetch_json(self, service: str, url: str, *, key_env: str | None = None, **params: Any) -> Any:
        r = self.net(url, **params)
        if r["status"] >= 400:
            raise PalError(explain(service, r["status"], r.get("text", ""), key_env))
        try:
            return json.loads(r["text"])
        except ValueError as e:
            raise PalError(f"{service} returned something that is not JSON") from e

    def rapid_host(self, tool: str, default: str) -> str:
        host = str(self.cfg.rapidapi_hosts.get(tool, default)).strip().lower()
        if not _RAPID_HOST.match(host):
            raise PalError(f"tools.rapidapi_hosts[{tool}] must be a RapidAPI host such as {default}")
        return host

    def hf_model(self, tool: str, default: str) -> str:
        model = str(self.cfg.hf_models.get(tool, default)).strip()
        # each part starts with a letter or digit, so ".." can never walk up the address that carries the token
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", model):
            raise PalError(f"tools.hf_models[{tool}] is not a valid model name")
        return model

    def media(self, kind: str) -> list[str]:
        exts = IMAGE_TYPES if kind == "image" else AUDIO_TYPES
        found: list[str] = []
        for prefix in MEDIA_DIRS:
            found += [p for p in self._os("file_list", {"prefix": prefix}) if "." in p and "." + p.rsplit(".", 1)[1].lower() in exts]
        return found

    def media_path(self, path: Any, kind: str) -> tuple[str, str]:
        """(path, content type) for a media file this session holds. Only uploads and generated files, only the right kind."""
        p = str(path).strip()
        exts = IMAGE_TYPES if kind == "image" else AUDIO_TYPES
        ext = "." + p.rsplit(".", 1)[1].lower() if "." in p else ""
        if not p.startswith(MEDIA_DIRS) or ext not in exts:
            have = ", ".join(self.media(kind)) or "none yet"
            raise PalError(f"give the path of an uploaded or generated {kind} file (for example uploads/name{next(iter(exts))}); available: {have}")
        return p, exts[ext]


# -- when an answer is in a shape that is not recognised, show it trimmed instead of failing --------------------------------------------------
def excerpt(data: Any, n: int = 700) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= n else text[:n] + " ..."
