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
from .trading import backtest, cost_sensitivity
from .paper import record_forecasts, settle_forecasts, read_paper
from .benchmark import (load_tfns, load_fintweet_2025, run_benchmark, prepare_annotations,
                        run_stock_movement_text_track, import_equinxx_tweets)
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
    backtest_cmd = commands.add_parser("backtest", help="backtest saved historical forecasts")
    backtest_cmd.add_argument("--run", type=Path, required=True); backtest_cmd.add_argument("--output", type=Path, required=True)
    backtest_cmd.add_argument("--model", default="linear"); backtest_cmd.add_argument("--feature-set", default="price_sentiment")
    backtest_cmd.add_argument("--threshold", type=float, default=0.0); backtest_cmd.add_argument("--cost-bps", type=float, default=5.0); backtest_cmd.add_argument("--slippage-bps", type=float, default=5.0)
    paper_record = commands.add_parser("paper-record", help="append eligible forward forecasts to the paper ledger")
    paper_record.add_argument("--forecasts", type=Path, required=True); paper_record.add_argument("--db", type=Path, required=True)
    paper_record.add_argument("--threshold", type=float, default=0.0); paper_record.add_argument("--cost-bps", type=float, default=5.0); paper_record.add_argument("--slippage-bps", type=float, default=5.0)
    paper_settle = commands.add_parser("paper-settle", help="settle due paper forecasts against canonical prices")
    paper_settle.add_argument("--prices", type=Path, required=True); paper_settle.add_argument("--db", type=Path, required=True)
    paper_status = commands.add_parser("paper-status", help="inspect pending and settled paper forecasts")
    paper_status.add_argument("--db", type=Path, required=True)
    benchmark = commands.add_parser("benchmark-sentiment", help="benchmark published or local sentiment labels")
    benchmark_source = benchmark.add_mutually_exclusive_group(required=True)
    benchmark_source.add_argument("--source", choices=["tfns"], help="explicitly download the public TFNS dataset")
    benchmark_source.add_argument("--input", type=Path, help="local CSV with text and label columns")
    benchmark.add_argument("--text-column", default="text"); benchmark.add_argument("--label-column", default="label")
    benchmark.add_argument("--split-column", default="split"); benchmark.add_argument("--id-column", default="post_id")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--label-provenance", choices=["user_supplied", "published", "pilot"], default="user_supplied")
    benchmark.add_argument("--annotation-metadata", type=Path, help="JSON attestation with anonymized IDs, qualifications, agreement, and adjudication evidence")
    benchmark.add_argument("--models", default="majority,vader,tfidf_logreg", help="comma-separated: majority,vader,tfidf_logreg,finbert,twitter_roberta")
    benchmark.add_argument("--seed", type=int, default=42); benchmark.add_argument("--validation-fraction", type=float, default=0.2)
    benchmark.add_argument("--batch-size", type=int, default=16); benchmark.add_argument("--cache-dir", type=Path, default=Path("artifacts/sentiment-benchmark/cache"))
    annotations = commands.add_parser("prepare-sentiment-annotations", help="make a blinded two-annotator finance-direction pilot batch")
    annotations.add_argument("--posts", type=Path, required=True); annotations.add_argument("--output", type=Path, required=True)
    annotations.add_argument("--sample-size", type=int, default=500); annotations.add_argument("--seed", type=int, default=42)
    annotations.add_argument("--stratify-column", help="optional explicitly authorized sampling stratum column")
    movement = commands.add_parser("benchmark-stock-movement", help="audit and optionally evaluate the separate exploratory FinTweet 2025 stock-movement track")
    movement_source = movement.add_mutually_exclusive_group(required=True)
    movement_source.add_argument("--source", choices=["fintweet-2025"], help="explicitly download michael-zus/fintweet-sentiment-2025")
    movement_source.add_argument("--input", type=Path, help="local CSV with timestamp, text, split, and label columns")
    movement.add_argument("--text-column", default="text"); movement.add_argument("--label-column", default="label_1d_3class")
    movement.add_argument("--split-column", default="split"); movement.add_argument("--timestamp-column", default="timestamp")
    movement.add_argument("--url-column", default="tweet_url"); movement.add_argument("--output", type=Path, required=True)
    movement.add_argument("--models", default="majority,tfidf_logreg", help="comma-separated text-only baselines: majority,vader,tfidf_logreg")
    movement.add_argument("--seed", type=int, default=42)
    equinxx = commands.add_parser("import-equinxx-tweets", help="map a local Equinxx stock_tweets.csv to canonical posts")
    equinxx.add_argument("--tweets", type=Path, required=True); equinxx.add_argument("--prices", type=Path, help="optional local stock_yfinance_data.csv")
    equinxx.add_argument("--output", type=Path, required=True)
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
        elif args.command == "backtest":
            input_path = args.run / "forecasts.csv" if args.run.is_dir() else args.run
            if not input_path.is_file():
                candidate = args.run / "forecast.csv"
                if candidate.is_file():
                    input_path = candidate
                else:
                    raise ValueError(f"forecast CSV not found in {args.run}")
            forecasts = pd.read_csv(input_path)
            strategy = {"model": args.model, "feature_set": args.feature_set, "threshold": args.threshold,
                        "cost_bps": args.cost_bps, "slippage_bps": args.slippage_bps}
            equity, trades, summary = backtest(forecasts, **strategy)
            sensitivity = cost_sensitivity(forecasts, model=args.model, feature_set=args.feature_set,
                threshold=args.threshold, slippage_bps=args.slippage_bps)
            args.output.mkdir(parents=True, exist_ok=True)
            equity.to_csv(args.output / "equity.csv", index=False)
            trades.to_csv(args.output / "trades.csv", index=False)
            sensitivity.to_csv(args.output / "cost_sensitivity.csv", index=False)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            result = {"output": str(args.output), "trade_rows": len(trades)}
        elif args.command == "paper-record":
            count = record_forecasts(args.db, pd.read_csv(args.forecasts), threshold=args.threshold,
                                     cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
            result = {"output": str(args.db), "recorded_rows": count}
        elif args.command == "paper-settle":
            prices = pd.read_csv(args.prices)
            count = settle_forecasts(args.db, prices)
            result = {"output": str(args.db), "settled_rows": count}
        elif args.command == "paper-status":
            records = read_paper(args.db)
            result = {"output": str(args.db), "pending_rows": int(records.status.eq("pending").sum()) if len(records) else 0,
                      "settled_rows": int(records.status.eq("settled").sum()) if len(records) else 0}
        elif args.command == "benchmark-sentiment":
            if args.source == "tfns":
                labelled, source_metadata = load_tfns()
            else:
                labelled = pd.read_csv(args.input)
                rename = {args.text_column: "text", args.label_column: "label"}
                if args.split_column in labelled: rename[args.split_column] = "split"
                if args.id_column in labelled: rename[args.id_column] = "post_id"
                labelled = labelled.rename(columns=rename)
                source_metadata = {"dataset_id": "local-labeled-input", "annotation_provenance": args.label_provenance}
                if args.annotation_metadata:
                    try:
                        source_metadata.update(json.loads(args.annotation_metadata.read_text(encoding="utf-8")))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(f"could not read annotation metadata: {exc}") from exc
            result = run_benchmark(labelled, args.output, models=[model.strip() for model in args.models.split(",") if model.strip()],
                seed=args.seed, validation_fraction=args.validation_fraction, batch_size=args.batch_size,
                cache_dir=args.cache_dir, dataset_metadata=source_metadata, input_path=args.input if args.input else None)
        elif args.command == "prepare-sentiment-annotations":
            posts = pd.read_csv(args.posts)
            result = prepare_annotations(posts, args.output, sample_size=args.sample_size, seed=args.seed, stratum_column=args.stratify_column)
        elif args.command == "benchmark-stock-movement":
            if args.source == "fintweet-2025":
                movement_data, source_metadata = load_fintweet_2025()
            else:
                movement_data = pd.read_csv(args.input)
                rename = {args.text_column: "text", args.label_column: "label_1d_3class", args.split_column: "split", args.timestamp_column: "timestamp"}
                if args.url_column in movement_data: rename[args.url_column] = "tweet_url"
                movement_data = movement_data.rename(columns=rename)
                source_metadata = {"dataset_id": str(args.input), "dataset_sha256": _sha256_file(args.input), "source_provenance": "local user-supplied input"}
            result = run_stock_movement_text_track(movement_data, args.output,
                models=[model.strip() for model in args.models.split(",") if model.strip()], seed=args.seed, dataset_metadata=source_metadata)
        elif args.command == "import-equinxx-tweets":
            result = import_equinxx_tweets(args.tweets, args.output, prices_path=args.prices)
        else:
            posts = collect_x(args.query, args.start, args.end, args.ticker, args.output, archive=args.archive, max_pages=args.max_pages); result = {"output": str(args.output), "post_rows": len(posts)}
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 1
    summary = [f"output={result['output']}"]
    for field in ("price_rows", "post_rows", "forecast_rows", "metric_rows", "prediction_rows", "annotation_rows", "audit_issue_count", "trade_rows", "recorded_rows", "settled_rows", "pending_rows"):
        if field in result: summary.append(f"{field}={result[field]}")
    print(" ".join(summary))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
