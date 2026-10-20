"""`sdk.os`: friendly names for the OS-kernel queries. Each call is an OSQuery routed to the OS kernel; a refusal
raises OSKernelError instead of returning an error value."""
from __future__ import annotations

from typing import Any

from pal.os_kernel import OSKernelError
from pal.sdk.queries import OSQuery


class OSAPI:
    def __init__(self, sdk: Any):
        self._sdk = sdk

    def _call(self, op: str, **params: Any) -> Any:
        resp = self._sdk.send_request(OSQuery(op=op, params=params))
        if not resp.ok:
            raise OSKernelError(resp.error)
        return resp.data

    def read_file(self, path: str) -> str:
        return self._call("file_read", path=path)

    def write_file(self, path: str, content: str) -> dict:
        return self._call("file_write", path=path, content=content)

    def read_bytes(self, path: str) -> bytes:
        """A workspace file as bytes (images, audio, PDFs, ...), within the OS kernel's binary size limit."""
        import base64
        return base64.b64decode(self._call("file_read_b64", path=path)["b64"])

    def file_to_markdown(self, path: str) -> str:
        """Convert a PDF, Word, PowerPoint, Excel, EPUB, HTML or LaTeX file in the workspace to markdown, save it under converted/,
        and return a description with the beginning of the text."""
        from pal.errors import PalError
        from pal.input_converters.components.base import ConversionError
        from pal.input_converters.workspace import convert_workspace_file, describe
        try:
            r = convert_workspace_file(path, read_bytes=self.read_bytes, write_text=lambda p, t: self.write_file(p, t),
                                       exists=lambda p: p in self.list_files(p))
        except ConversionError as e:
            raise PalError(str(e)) from e
        return describe(r)

    def list_files(self, prefix: str = "") -> list[str]:
        return self._call("file_list", prefix=prefix)

    def delete_file(self, path: str) -> dict:
        return self._call("file_delete", path=path)

    def http_get(self, url: str) -> dict:
        return self._call("net_get", url=url)

    def http_post(self, url: str, json: Any = None) -> dict:
        return self._call("net_post", url=url, json=json)

    def system_info(self) -> dict:
        return self._call("sys_info")

    def keyboard(self, **kw: Any) -> Any:
        return self._call("keyboard", **kw)

    def mouse(self, **kw: Any) -> Any:
        return self._call("mouse", **kw)


def register_os_tools(sdk: Any) -> list[str]:
    """Make the OS operations callable as tools (for ReAct and similar agents). The tools run as OSQuery calls, so they
    reach the OS kernel and never a model. Not registered by default: the chat's tools stay read-only."""
    from pal.tool_manager.components.base import ToolParam

    tm = sdk._tool_manager
    api = sdk.os
    specs = [
        ("read_file", lambda: api.read_file, [ToolParam("path", str)]),
        ("write_file", lambda: api.write_file, [ToolParam("path", str), ToolParam("content", str)]),
        ("list_files", lambda: api.list_files, [ToolParam("prefix", str, required=False, default="")]),
        ("file_to_markdown", lambda: api.file_to_markdown, [ToolParam("path", str)]),
        ("http_get", lambda: api.http_get, [ToolParam("url", str)]),
        ("system_info", lambda: api.system_info, []),
    ]
    for name, loader, params in specs:
        tm.register_tool(name, loader, params=params)
    return [n for n, _, _ in specs]
