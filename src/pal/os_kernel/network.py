"""Network requests through the OS network stack, with the checks an agent-facing call needs.

http and https only; no credentials in the URL; the host must resolve to a public address unless the settings allow
private ones (blocks requests to localhost, the home network and cloud metadata endpoints); an optional host
allow-list; a timeout; a response size cap; redirects are followed by hand so every hop is checked again.
Residual risk, stated plainly: the address is checked before connecting, so a hostile DNS server that answers
differently the second time (DNS rebinding) is not fully covered.
"""
from __future__ import annotations

import contextlib
import ipaddress
import logging
import socket
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin, urlsplit

from pal.os_kernel.filesystem import OSKernelError

_MAX_REDIRECTS = 3


def check_url(url: str, allow_private: bool, allowed_hosts: list[str]) -> None:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise OSKernelError("only http and https URLs are allowed")
    if p.username or p.password:
        raise OSKernelError("credentials inside a URL are not allowed")
    host = p.hostname.lower()
    if allowed_hosts and host not in {h.lower() for h in allowed_hosts}:
        raise OSKernelError(f"host {host!r} is not on the allowed list")
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as e:
        raise OSKernelError(f"could not resolve {host!r}") from e
    if allow_private:
        return
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise OSKernelError(f"{host!r} resolves to a private or reserved address, which is blocked")


# Headers a caller may set. Everything else (Host, Cookie, Proxy-*, X-Forwarded-*, ...) is refused, so a request cannot be made
# to look like it came from somewhere else. These are the ones service APIs use for their keys.
SAFE_HEADERS = frozenset({"authorization", "x-api-key", "x-rapidapi-key", "x-rapidapi-host", "ocp-apim-subscription-key",
                          "x-goog-api-key", "x-goog-fieldmask", "content-type", "accept"})
_TEXT_TYPES = ("application/json", "application/xml", "application/atom+xml", "application/xhtml+xml", "application/javascript")


def clean_headers(headers: Any) -> dict:
    out: dict = {}
    for k, v in (headers or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise OSKernelError("header names and values must be text")
        if k.lower() not in SAFE_HEADERS:
            raise OSKernelError(f"header {k!r} is not allowed")
        if len(v) > 2000 or any(c in v for c in "\r\n\x00"):
            raise OSKernelError(f"header {k!r} has an invalid value")
        out[k] = v
    return out


def is_text(content_type: str) -> bool:
    ct = content_type.split(";")[0].strip().lower()
    return not ct or ct.startswith("text/") or ct in _TEXT_TYPES or ct.endswith(("+json", "+xml"))


def safe_target(url: str) -> str:
    """The URL without its query string, fragment or credentials: what is written to logs, because the query may hold an API key."""
    p = urlsplit(url)
    if not (p.scheme and p.hostname):
        return ""
    return f"{p.scheme}://{p.hostname}{f':{p.port}' if p.port else ''}{p.path}"


_quiet_lock = threading.Lock()
_quiet_depth = 0
_quiet_saved = logging.NOTSET


@contextlib.contextmanager
def _quiet_http_logs() -> Iterator[None]:
    """urllib3's DEBUG log line holds the full request URL, query string included, and so would put an API key in a log file for
    anyone running with debug logging. Silence it for the length of a request (counted, so overlapping requests are safe)."""
    global _quiet_depth, _quiet_saved
    lg = logging.getLogger("urllib3")
    with _quiet_lock:
        if _quiet_depth == 0:
            _quiet_saved = lg.level
            lg.setLevel(logging.WARNING)
        _quiet_depth += 1
    try:
        yield
    finally:
        with _quiet_lock:
            _quiet_depth -= 1
            if _quiet_depth == 0:
                lg.setLevel(_quiet_saved)


def request(method: str, url: str, *, allow_private: bool, allowed_hosts: list[str], timeout: float, max_bytes: int,
            json: Any = None, headers: dict | None = None, data: bytes | None = None, max_binary_bytes: int | None = None,
            keep_binary: bool = False) -> dict:
    """One HTTP request with the checks above. `headers` must already be cleaned. A text answer comes back as `text` (at most
    `max_bytes`); with `keep_binary` any other answer comes back as `body` bytes (at most `max_binary_bytes`) and `text` is empty.
    Redirects are followed by hand and the caller's headers are NOT sent to the new address (they hold keys)."""
    with _quiet_http_logs():
        return _request(method, url, allow_private, allowed_hosts, timeout, max_bytes, json, headers, data, max_binary_bytes, keep_binary)


def _request(method: str, url: str, allow_private: bool, allowed_hosts: list[str], timeout: float, max_bytes: int, json: Any,
             headers: dict | None, data: bytes | None, max_binary_bytes: int | None, keep_binary: bool) -> dict:
    import requests

    sent = dict(headers or {})
    for _ in range(_MAX_REDIRECTS + 1):
        check_url(url, allow_private, allowed_hosts)
        try:
            resp = requests.request(method, url, json=json, data=data, timeout=timeout, allow_redirects=False, stream=True,
                                    headers={"User-Agent": "pal-code", **sent})
        except requests.RequestException as e:
            raise OSKernelError(f"request failed: {type(e).__name__}") from e
        if resp.is_redirect and resp.headers.get("Location"):
            url, json, data, method, sent = urljoin(url, resp.headers["Location"]), None, None, "GET", {}
            resp.close()
            continue
        ctype = resp.headers.get("Content-Type", "")
        textual = is_text(ctype) or not keep_binary
        cap = max_bytes if textual else (max_binary_bytes or max_bytes)
        body = b""
        for chunk in resp.iter_content(8192):
            body += chunk
            if len(body) > cap:
                body = body[:cap]
                break
        resp.close()
        out = {"status": resp.status_code, "content_type": ctype, "truncated": len(body) >= cap, "bytes": len(body)}
        if textual:
            out["text"] = body.decode(resp.encoding or "utf-8", errors="replace")
        else:
            out["text"], out["body"] = "", body
        return out
    raise OSKernelError("too many redirects")
