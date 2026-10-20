"""One function per command-line command. Each takes the engine and the parsed arguments and returns an exit code."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from localrag.engine import LocalRAG
from localrag.eval.golden import load_golden, save_golden
from localrag.eval.report import format_report
from localrag.eval.runner import evaluate
from localrag.eval.synthetic import generate_golden
from localrag.mcp.factory import build_server
from localrag.mcp.transport_stdio import serve_stdio


def cmd_ingest(engine: LocalRAG, args: Namespace) -> int:
    for path in args.paths:
        for item in engine.ingest_path(path, not args.no_recursive):
            print(f"{item['source']}: {item['chunks']} chunks ({item['chunker']}), {item['duplicates_dropped']} duplicates dropped")
    print(f"saved {engine.save()} bytes to {engine.snapshot_path()}")
    return 0


def cmd_query(engine: LocalRAG, args: Namespace) -> int:
    result = engine.search(args.text, args.top_k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    for hit in result.hits:
        print(f"{hit.rank}. {hit.chunk.metadata.get('source')} (score {hit.score:.3f}, {hit.retriever})")
        print("   " + hit.chunk.text[:200].replace("\n", " "))
    if not result.hits:
        print("no results")
    return 0 if result.hits else 1


def cmd_tree(engine: LocalRAG, args: Namespace) -> int:
    if args.document:
        print(engine.outline(args.document))
        return 0
    for m in engine.tree_search(args.text, args.top_n):
        print(f"{m['score']:.2f}  {m['path']}\n      {m['summary']}")
    return 0


def cmd_stats(engine: LocalRAG, args: Namespace) -> int:
    print(json.dumps(engine.stats(), indent=2))
    return 0


def cmd_list(engine: LocalRAG, args: Namespace) -> int:
    for d in engine.list_documents():
        print(f"{d['doc_id']}  {d['chunks']:>4} chunks  {d['source']}")
    return 0


def cmd_eval(engine: LocalRAG, args: Namespace) -> int:
    golden = load_golden(args.golden) if args.golden else generate_golden(engine, args.queries, args.seed)
    if args.save_golden:
        save_golden(args.save_golden, golden)
    print(format_report(evaluate(engine, golden, args.k)))
    return 0


def cmd_serve_mcp(engine: LocalRAG, args: Namespace) -> int:
    roots = [Path(r) for r in args.allow] or [Path.cwd()]
    serve_stdio(build_server(engine, roots))
    return 0
