"""Command line entry point for offline research and explicit data operations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from .data import collect_x, fetch_prices, import_original, recover_history, normalize_posts, normalize_prices
from .demo import make_demo
from .sentiment import score_posts
from .features import build_features
from .research import run_research, forecast_latest
import hashlib
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentiment-stock", description="Offline sentiment and stock research.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate and run a labelled synthetic research demo")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--include-models", action="store_true")
    generate = commands.add_parser("generate-demo", help="write reproducible synthetic canonical data")
    generate.add_argument("--output", type=Path, default=Path("data/demo"))
    generate.add_argument("--sessions", type=int, default=260)
    generate.add_argument("--seed", type=int, default=11)
    recovery = commands.add_parser("recover-history", help="recover known archived CSVs from git")
    recovery.add_argument("--repo", type=Path, default=Path.cwd()); recovery.add_argument("--output", type=Path, default=Path("data/recovered"))
    original = commands.add_parser("import-original", help="join original tweet export to ticker mapping")
    original.add_argument("--tweets", type=Path, required=True); original.add_argument("--mapping", type=Path, required=True); original.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("fetch-prices", help="explicitly fetch adjusted Yahoo Finance bars")
    fetch.add_argument("--tickers", nargs="+", required=True); fetch.add_argument("--start", required=True); fetch.add_argument("--end", required=True); fetch.add_argument("--output", type=Path, default=Path("data/fetched"))
    collect = commands.add_parser("collect-x", help="explicitly collect bounded X API v2 search pages")
    collect.add_argument("--query", required=True); collect.add_argument("--start", required=True); collect.add_argument("--end", required=True); collect.add_argument("--ticker", required=True); collect.add_argument("--output", type=Path, default=Path("data/x_collection")); collect.add_argument("--archive", action="store_true"); collect.add_argument("--max-pages", type=int, default=10)
    run = commands.add_parser("run", help="score canonical inputs, build causal features, and run walk-forward research")
    run.add_argument("--prices", type=Path, required=True); run.add_argument("--posts", type=Path, required=True); run.add_argument("--output", type=Path, required=True)
    run.add_argument("--target", choices=["return", "next_close", "volatility"], default="return"); run.add_argument("--encoder", choices=["provided", "vader", "finbert", "twitter"], default="provided")
    run.add_argument("--include-models", action="store_true"); run.add_argument("--splits", type=int, default=3); run.add_argument("--seed", type=int, default=11); run.add_argument("--events", type=Path)
    forecast = commands.add_parser("forecast", help="record an offline forward forecast from canonical input files")
    forecast.add_argument("--prices", type=Path, required=True); forecast.add_argument("--posts", type=Path, required=True); forecast.add_argument("--output", type=Path, required=True); forecast.add_argument("--as-of", required=True)
    forecast.add_argument("--encoder", choices=["provided", "vader", "finbert", "twitter"], default="provided"); forecast.add_argument("--model", default="linear"); forecast.add_argument("--feature-set", default="price_sentiment"); forecast.add_argument("--seed", type=int, default=11)
    return parser


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _inputs(prices_path: Path, posts_path: Path, encoder: str):
    prices = normalize_prices(pd.read_csv(prices_path))
    posts, encoder_metadata = score_posts(normalize_posts(pd.read_csv(posts_path)), encoder)
    dataset_path = prices_path.parent / "dataset.json"
    metadata = json.loads(dataset_path.read_text(encoding="utf-8")) if dataset_path.exists() else {"dataset_type": "user_supplied", "prices_path": str(prices_path), "posts_path": str(posts_path)}
    metadata["encoder"] = encoder_metadata
    sample_groups = []
    for _, group in posts.groupby("ticker", sort=True):
        positions = sorted(set(np.linspace(0, len(group) - 1, min(4, len(group)), dtype=int))) if len(group) else []
        sample_groups.append(group.iloc[positions])
    sample = pd.concat(sample_groups, ignore_index=True) if sample_groups else posts.head(0)
    metadata["posts_sample"] = sample[[column for column in ("post_id", "ticker", "created_at", "available_at", "text", "author", "sentiment_score", "source") if column in sample]].to_dict("records")
    metadata["input_hashes"] = {"prices_sha256": _sha256_file(prices_path), "posts_sha256": _sha256_file(posts_path)}
    return prices, posts, metadata


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-demo":
            result = make_demo(args.output, sessions=args.sessions, seed=args.seed)
        elif args.command == "demo":
            data_dir = args.output / "data"
            make_demo(data_dir)
            prices, posts, metadata = _inputs(data_dir / "prices.csv", data_dir / "posts.csv", "provided")
            result = run_research(build_features(prices, posts), args.output, include_models=args.include_models, dataset_metadata=metadata)
        elif args.command == "recover-history":
            result = recover_history(args.repo, args.output)
        elif args.command == "import-original":
            result = import_original(args.tweets, args.mapping, args.output)
        elif args.command == "fetch-prices":
            prices = fetch_prices(args.tickers, args.start, args.end, args.output); result = {"output": str(args.output), "price_rows": len(prices)}
        elif args.command == "run":
            prices, posts, metadata = _inputs(args.prices, args.posts, args.encoder)
            events = pd.read_csv(args.events) if args.events else None
            built = build_features(prices, posts, target=args.target, events=events)
            result = run_research(built, args.output, seed=args.seed, n_splits=args.splits, include_models=args.include_models, dataset_metadata=metadata)
        elif args.command == "forecast":
            prices, posts, metadata = _inputs(args.prices, args.posts, args.encoder)
            forecast = forecast_latest(prices, posts, args.output, as_of=pd.Timestamp(args.as_of), model=args.model, feature_set=args.feature_set, seed=args.seed)
            forecast_metadata_path = args.output / "metadata.json"
            forecast_metadata = json.loads(forecast_metadata_path.read_text(encoding="utf-8"))
            forecast_metadata["dataset_metadata"] = metadata
            forecast_metadata_path.write_text(json.dumps(forecast_metadata, indent=2, default=str) + "\n", encoding="utf-8")
            result = {"output": str(args.output), "forecast_rows": len(forecast)}
        else:
            posts = collect_x(args.query, args.start, args.end, args.ticker, args.output, archive=args.archive, max_pages=args.max_pages); result = {"output": str(args.output), "post_rows": len(posts)}
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 1
    summary = [f"output={result['output']}"]
    for field in ("price_rows", "post_rows", "forecast_rows", "metric_rows"):
        if field in result: summary.append(f"{field}={result[field]}")
    print(" ".join(summary))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
