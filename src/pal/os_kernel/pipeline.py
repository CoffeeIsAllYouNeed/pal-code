"""The OS kernel interface: where non-LLM queries go.

An agent's non-LLM requests (file operations, network requests, system facts) are not model work and never touch the
LLM. They are system calls into the operating system, here Ubuntu's Linux kernel: files through its filesystem, requests
through its network stack, facts through /proc. This class is the guarded doorway: a per-agent workspace, size and
network limits, the access manager's confirmation gate for destructive operations, and an audit trail (operation and
target only, never file contents).
"""
from __future__ import annotations

import base64
import logging
from collections import deque
from pathlib import Path
from typing import Any

from pal.access_manager.components.base import OperationType
from pal.access_manager.pipeline import AccessManagerPipeline
from pal.config import OsKernelConfig
from pal.os_kernel import network
from pal.os_kernel.filesystem import OSKernelError, Workspace
from pal.os_kernel.sysinfo import system_info
from pal.paths import data_dir, safe_id

log = logging.getLogger(__name__)

OS_OPS = ("file_read", "file_write", "file_list", "file_delete", "file_read_b64", "net_get", "net_post", "net_request", "sys_info",
          "keyboard", "mouse")

# the file extension for a downloaded media type, used when a caller saves a response as "name.{ext}"
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "audio/flac": "flac",
        "audio/x-flac": "flac", "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mpeg": "mp3",
        "audio/ogg": "ogg", "audio/mp4": "m4a"}


class OSKernel:
    def __init__(self, cfg: OsKernelConfig | dict | None = None, access: AccessManagerPipeline | None = None,
                 root: Path | None = None):
        self.cfg = OsKernelConfig.coerce(cfg)
        self.access = access or AccessManagerPipeline()
        self.root = Path(root) if root else (Path(self.cfg.workspace_dir).expanduser() if self.cfg.workspace_dir
                                             else data_dir("workspaces"))
        self._workspaces: dict[str, Workspace] = {}
        self.audit: deque[dict] = deque(maxlen=500)

    def workspace(self, agent_id: str) -> Workspace:
        key = safe_id(agent_id)
        if key not in self._workspaces:
            self._workspaces[key] = Workspace(self.root / key, self.cfg.max_read_bytes, self.cfg.max_write_bytes,
                                              self.cfg.max_binary_bytes)
        return self._workspaces[key]

    def drop_workspace(self, agent_id: str) -> bool:
        """Delete an agent's whole workspace: used when a chat session ends, so uploaded and generated files do not outlive it."""
        import shutil
        key = safe_id(agent_id)
        self._workspaces.pop(key, None)
        path = self.root / key
        existed = path.is_dir()
        shutil.rmtree(path, ignore_errors=True)
        return existed

    def prune_stale(self, prefix: str, max_age_seconds: float) -> int:
        """Delete workspaces whose folder name starts with `prefix` and that nothing has touched for `max_age_seconds` (left behind
        by a crash or an upload whose chat never started). Other workspaces, such as an SDK agent's own, are not touched."""
        import shutil
        import time
        removed = 0
        if not self.root.is_dir():
            return 0
        for d in self.root.iterdir():
            try:
                if d.is_dir() and d.name.startswith(prefix) and time.time() - self._last_touched(d) > max_age_seconds:
                    self._workspaces.pop(d.name, None)
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    @staticmethod
    def _last_touched(d: Path) -> float:
        times = [d.stat().st_mtime] + [p.stat().st_mtime for p in d.rglob("*") if p.is_file()]
        return max(times)

    def _authorize(self, agent_id: str, op: OperationType, resource: str) -> None:
        decision = self.access.authorize(agent_id, agent_id, op, resource=resource)
        if not decision.allowed:
            raise OSKernelError(f"{op.value} of {resource!r} was not permitted: {decision.reason or 'denied'}")

    def handle(self, agent_id: str, op: str, params: dict | None = None) -> Any:
        params = params or {}
        # a URL is logged without its query string: that is where a service key may be
        target = network.safe_target(str(params["url"])) if params.get("url") else str(params.get("path") or params.get("prefix") or "")
        ok = False
        try:
            if not self.cfg.enabled:
                raise OSKernelError("OS access is turned off in the settings")
            if op not in OS_OPS:
                raise OSKernelError(f"unknown OS operation {op!r}; available: {', '.join(OS_OPS)}")
            result = self._dispatch(agent_id, op, params)
            ok = True
            return result
        finally:
            self.audit.append({"agent": agent_id, "op": op, "target": target[:200], "ok": ok})
            log.info("os call agent=%s op=%s target=%s ok=%s", agent_id, op, target[:120], ok)

    def _dispatch(self, agent_id: str, op: str, p: dict) -> Any:
        if op in ("keyboard", "mouse"):
            raise OSKernelError("keyboard and mouse control need a desktop session and are not supported here")
        if op == "sys_info":
            return system_info()
        if op in ("net_get", "net_post"):
            return network.request(
                "GET" if op == "net_get" else "POST", str(p.get("url", "")), allow_private=self.cfg.allow_private_network,
                allowed_hosts=list(self.cfg.allowed_hosts), timeout=self.cfg.net_timeout_seconds,
                max_bytes=self.cfg.net_max_bytes, json=p.get("json"))
        if op == "net_request":
            return self._net_request(agent_id, p)
        ws = self.workspace(agent_id)
        if op == "file_list":
            return ws.list(str(p.get("prefix", "")))
        path = str(p.get("path", ""))
        ws.resolve(path)                      # validate first: nobody is ever asked to confirm an invalid path
        if op == "file_read":
            return ws.read(path)
        if op == "file_read_b64":
            data = ws.read_bytes(path)
            return {"path": path, "bytes": len(data), "b64": base64.b64encode(data).decode("ascii")}
        if op == "file_write":
            self._authorize(agent_id, OperationType.OVERWRITE if ws.exists(path) else OperationType.WRITE, path)
            return {"path": path, "bytes": ws.write(path, p.get("content", ""))}
        if not ws.exists(path):               # file_delete: nothing to confirm if there is nothing to delete
            raise OSKernelError(f"no such file: {path}")
        self._authorize(agent_id, OperationType.DELETE, path)
        ws.delete(path)
        return {"path": path, "deleted": True}

    def _net_request(self, agent_id: str, p: dict) -> dict:
        """A network request with auth headers (from an allow-list), and binary data in either direction through the workspace:
        `body_path` sends a workspace file as the request body, `save_as` stores a non-text answer in the workspace (a literal
        "{ext}" in the name becomes the right extension). Text answers come back as `text`."""
        method = str(p.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            raise OSKernelError("only GET and POST requests are supported")
        ws = self.workspace(agent_id)
        headers = network.clean_headers(p.get("headers"))
        data = None
        if p.get("body_path"):
            if p.get("json") is not None:
                raise OSKernelError("send either a JSON body or a file, not both")
            data = ws.read_bytes(str(p["body_path"]))
            headers["Content-Type"] = str(p.get("content_type") or "application/octet-stream")
            headers = network.clean_headers(headers)
        save_as = str(p["save_as"]) if p.get("save_as") else ""
        if save_as:
            ws.resolve(save_as.replace("{ext}", "bin"))           # validate the name before spending a request
        res = network.request(method, str(p.get("url", "")), allow_private=self.cfg.allow_private_network,
                              allowed_hosts=list(self.cfg.allowed_hosts), timeout=self.cfg.net_timeout_seconds,
                              max_bytes=self.cfg.net_max_bytes, json=p.get("json"), headers=headers, data=data,
                              max_binary_bytes=self.cfg.max_binary_bytes, keep_binary=True)
        body = res.pop("body", None)
        if body is None:
            return res
        res["binary"] = True
        if res["truncated"]:
            raise OSKernelError(f"the answer is larger than the {self.cfg.max_binary_bytes:,}-byte limit")
        if not save_as or res["status"] >= 400:
            return {**res, "saved": ""}
        ext = _EXT.get(res["content_type"].split(";")[0].strip().lower(), "bin")
        path = save_as.replace("{ext}", ext)
        ws.resolve(path)
        self._authorize(agent_id, OperationType.OVERWRITE if ws.exists(path) else OperationType.WRITE, path)
        ws.write_bytes(path, body)
        return {**res, "saved": path}
