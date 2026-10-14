"""Command line entry point for explicit data operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import collect_x, fetch_prices, import_original, recover_history
from .demo import make_demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentiment-stock", description="Prepare canonical sentiment and stock-price datasets.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("generate-demo", help="write reproducible synthetic data")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    demo.add_argument("--sessions", type=int, default=260)
    demo.add_argument("--seed", type=int, default=11)
    recovery = commands.add_parser("recover-history", help="recover known archived CSVs from git")
    recovery.add_argument("--repo", type=Path, default=Path.cwd())
    recovery.add_argument("--output", type=Path, default=Path("data/recovered"))
    original = commands.add_parser("import-original", help="join original tweet export to ticker mapping")
    original.add_argument("--tweets", type=Path, required=True)
    original.add_argument("--mapping", type=Path, required=True)
    original.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("fetch-prices", help="explicitly fetch adjusted Yahoo Finance bars")
    fetch.add_argument("--tickers", nargs="+", required=True)
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", type=Path, default=Path("data/fetched"))
    collect = commands.add_parser("collect-x", help="explicitly collect bounded X API v2 search pages")
    collect.add_argument("--query", required=True)
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    collect.add_argument("--ticker", required=True)
    collect.add_argument("--output", type=Path, default=Path("data/x_collection"))
    collect.add_argument("--archive", action="store_true")
    collect.add_argument("--max-pages", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-demo":
            result = make_demo(args.output, sessions=args.sessions, seed=args.seed)
        elif args.command == "recover-history":
            result = recover_history(args.repo, args.output)
        elif args.command == "import-original":
            result = import_original(args.tweets, args.mapping, args.output)
        elif args.command == "fetch-prices":
            prices = fetch_prices(args.tickers, args.start, args.end, args.output)
            result = {"output": str(args.output), "price_rows": len(prices)}
        else:
            posts = collect_x(args.query, args.start, args.end, args.ticker, args.output, archive=args.archive, max_pages=args.max_pages)
            result = {"output": str(args.output), "post_rows": len(posts)}
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = [f"output={result['output']}"]
    if "price_rows" in result:
        summary.append(f"price_rows={result['price_rows']}")
    if "post_rows" in result:
        summary.append(f"post_rows={result['post_rows']}")
    print(" ".join(summary))
    return 0
