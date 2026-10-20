"""The stdio transport: one JSON message per line on standard input, one reply per line on standard output."""
from __future__ import annotations

import sys
from typing import TextIO

from localrag.core.log import get_logger
from localrag.mcp.server import McpServer

log = get_logger(__name__)


def serve_stdio(server: McpServer, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Read lines until the input ends, answering each one. Returns how many lines were handled. Logs go to stderr, never stdout."""
    source, sink = stdin or sys.stdin, stdout or sys.stdout
    handled = 0
    for line in source:
        line = line.strip()
        if not line:
            continue
        handled += 1
        reply = server.handle_raw(line)
        if reply is not None:
            sink.write(reply + "\n")
            sink.flush()
    log.debug("stdin closed after %d messages", handled)
    return handled
