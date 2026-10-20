"""The command-line interface: ingest files, query, browse by section, evaluate, and serve over MCP."""
from __future__ import annotations

import argparse
import sys

from localrag import __version__
from localrag.cli import commands
from localrag.core.config import load_config
from localrag.core.errors import LocalRAGError
from localrag.engine import LocalRAG


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localrag", description="A local-first RAG engine and MCP server.")
    p.add_argument("--version", action="version", version=f"localrag {__version__}")
    p.add_argument("--config", help="path to a JSON settings file")
    p.add_argument("--data-dir", help="where the snapshot is kept (default .localrag)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("ingest", help="index files or folders")
    s.add_argument("paths", nargs="+")
    s.add_argument("--no-recursive", action="store_true")
    s = sub.add_parser("query", help="search the index")
    s.add_argument("text")
    s.add_argument("-k", "--top-k", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("tree", help="browse by section: search sections, or print a document outline")
    s.add_argument("text", nargs="?", default="")
    s.add_argument("--document", help="print the outline of this document instead of searching")
    s.add_argument("--top-n", type=int, default=5)
    sub.add_parser("stats", help="show index statistics")
    sub.add_parser("list", help="list indexed documents")
    s = sub.add_parser("eval", help="measure retrieval quality")
    s.add_argument("--golden", help="a golden set file; by default queries are generated from the index")
    s.add_argument("--save-golden", help="write the golden set used to this file")
    s.add_argument("--queries", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-k", type=int, default=5)
    s = sub.add_parser("serve-mcp", help="run as an MCP server on stdin and stdout")
    s.add_argument("--allow", action="append", default=[], help="a folder the ingest_path tool may read (repeatable)")
    return p


_COMMANDS = {"ingest": commands.cmd_ingest, "query": commands.cmd_query, "tree": commands.cmd_tree, "stats": commands.cmd_stats,
             "list": commands.cmd_list, "eval": commands.cmd_eval, "serve-mcp": commands.cmd_serve_mcp}


def open_engine(config_path: str | None, data_dir: str | None) -> LocalRAG:
    """The engine, restored from the snapshot in the data directory when there is one."""
    config = load_config(config_path)
    if data_dir:
        config.data_dir = data_dir
    engine = LocalRAG(config)
    snapshot = engine.snapshot_path()
    return LocalRAG.load(snapshot) if snapshot.is_file() else engine


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](open_engine(args.config, args.data_dir), args)
    except LocalRAGError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
