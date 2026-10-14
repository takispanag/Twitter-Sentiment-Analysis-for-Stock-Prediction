"""Canonical data import, recovery, and explicitly triggered collection."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

RECOVERY_COMMIT = "fc6a3730cdc85021f7c0a0a3756b8c9c06b38eec"
TICKERS = {"amazon": "AMZN", "apple": "AAPL", "google": "GOOG", "microsoft": "MSFT", "tesla": "TSLA"}
PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "market_open", "market_close"]
POST_COLUMNS = ["post_id", "ticker", "created_at", "available_at", "text", "author", "sentiment_score", "source", "time_precision"]
_NYSE = mcal.get_calendar("NYSE")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rename_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    folded = {str(column).strip().casefold().replace(" ", "_"): column for column in frame.columns}
    rename = {}
    for alias, canonical in aliases.items():
        original = folded.get(alias.casefold().replace(" ", "_"))
        if original is not None and canonical not in frame.columns:
            rename[original] = canonical
    return frame.rename(columns=rename).copy()


def _utc_timestamp(value: Any, *, field: str, allow_date: bool = False) -> tuple[pd.Timestamp, bool]:
    if pd.isna(value):
        raise ValueError(f"{field} contains a missing timestamp")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} contains an invalid epoch")
        stamp = pd.to_datetime(value, unit="s", utc=True)
        return stamp, False
    if isinstance(value, pd.Timestamp):
        stamp = value
        is_date = stamp.tz is None and stamp == stamp.normalize()
    elif isinstance(value, datetime):
        stamp = pd.Timestamp(value)
        is_date = stamp.tzinfo is None and stamp.time() == datetime.min.time()
    elif isinstance(value, date):
        stamp = pd.Timestamp(value)
        is_date = True
    else:
        text = str(value).strip()
        is_date = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))
        try:
            stamp = pd.Timestamp(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} contains an invalid timestamp: {text!r}") from exc
    if stamp.tzinfo is None:
        if allow_date and is_date:
            return stamp.tz_localize("UTC"), True
        raise ValueError(f"{field} intraday timestamps must include a timezone")
    return stamp.tz_convert("UTC"), bool(allow_date and is_date)


def _utc_series(values: pd.Series, *, field: str, allow_date: bool = False) -> tuple[pd.Series, pd.Series]:
    """Parse mixed date-only, epoch, and explicitly zoned timestamps in batches."""
    strings = values.astype("string").str.strip()
    missing = values.isna() | strings.eq("")
    if missing.any():
        raise ValueError(f"{field} contains a missing timestamp")
    date_only = strings.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    midnight_naive = strings.str.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?")
    if allow_date:
        date_only = date_only | midnight_naive
    numeric = pd.to_numeric(strings, errors="coerce").notna() & ~date_only
    zoned = strings.str.contains(r"(?:Z|[+-]\d{2}(?::?\d{2})?)$", case=False, regex=True, na=False)
    # Python datetime/date objects keep their timezone even when their repr is unusual.
    object_zoned = values.map(lambda v: isinstance(v, (datetime, pd.Timestamp)) and v.tzinfo is not None)
    parse_timestamp = ~date_only & ~numeric
    if (parse_timestamp & ~zoned & ~object_zoned).any():
        raise ValueError(f"{field} intraday timestamps must include a timezone")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if date_only.any():
        parsed.loc[date_only] = pd.to_datetime(strings.loc[date_only].str.slice(0, 10), format="%Y-%m-%d", utc=True)
    if numeric.any():
        epoch = pd.to_numeric(strings.loc[numeric], errors="coerce")
        if not np.isfinite(epoch.to_numpy(dtype=float)).all():
            raise ValueError(f"{field} contains an invalid epoch")
        parsed.loc[numeric] = pd.to_datetime(epoch, unit="s", utc=True)
    if parse_timestamp.any():
        try:
            parsed.loc[parse_timestamp] = pd.to_datetime(strings.loc[parse_timestamp], utc=True, format="mixed")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} contains an invalid timestamp") from exc
    if parsed.isna().any():
        raise ValueError(f"{field} contains an invalid timestamp")
    return parsed, date_only.astype(bool)


def _ticker_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _rename_columns(frame, {"ticker_symbol": "ticker", "symbol": "ticker"})
    if "ticker" not in frame:
        raise ValueError("required column missing: ticker (or ticker_symbol)")
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    if frame["ticker"].isna().any() or (frame["ticker"] == "").any():
        raise ValueError("ticker values must be present")
    return frame


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate price bars and attach NYSE UTC session boundaries."""
    frame = _ticker_column(frame)
    frame = _rename_columns(frame, {"date": "date", "datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "adj_close": "adj_close", "volume": "volume"})
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"required price columns missing: {', '.join(sorted(missing))}")
    session_dates = []
    for value in frame["date"]:
        stamp, _ = _utc_timestamp(value, field="date", allow_date=True)
        day = stamp.date()
        if stamp.time() != datetime.min.time() and stamp != stamp.normalize():
            raise ValueError("price dates must identify an NYSE session date")
        session_dates.append(pd.Timestamp(day))
    frame["date"] = session_dates
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if "adj_close" in frame:
        adj = pd.to_numeric(frame["adj_close"], errors="coerce").to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        if not np.isfinite(adj).all() or (adj <= 0).any() or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("adjusted close and source close must be finite and positive")
        factor = adj / close
        if not np.isfinite(factor).all() or (factor <= 0).any():
            raise ValueError("adjustment factors must be finite and positive")
        with np.errstate(over="ignore", invalid="ignore"):
            for column in ("open", "high", "low", "close"):
                frame[column] = frame[column].to_numpy(dtype=float) * factor
    ohlc = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(ohlc).all():
        raise ValueError("adjusted OHLC values must be finite")
    if (ohlc <= 0).any():
        raise ValueError("OHLC values must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be nonnegative")
    if ((frame["low"] > frame[["open", "close"]].min(axis=1)) | (frame["high"] < frame[["open", "close"]].max(axis=1)) | (frame["low"] > frame["high"])).any():
        raise ValueError("invalid OHLC bounds")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/session price rows")

    dates = frame["date"].dt.strftime("%Y-%m-%d")
    schedule = _NYSE.schedule(start_date=dates.min(), end_date=dates.max())
    schedule.index = pd.to_datetime(schedule.index).tz_localize(None).normalize()
    if not frame["date"].isin(schedule.index).all():
        bad = frame.loc[~frame["date"].isin(schedule.index), "date"].iloc[0].date()
        raise ValueError(f"{bad} is not an NYSE session")
    frame["market_open"] = frame["date"].map(schedule["market_open"])
    frame["market_close"] = frame["date"].map(schedule["market_close"])
    return frame[PRICE_COLUMNS].sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def _id_string(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("post_id contains a missing identifier")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_posts(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize posts without guessing the timezone of naive intraday values."""
    frame = _ticker_column(frame)
    frame = _rename_columns(frame, {"tweet_id": "post_id", "id": "post_id", "post_date": "created_at", "created_at": "created_at", "observed_at": "observed_at", "availability_time": "observed_at", "body": "text", "content": "text", "writer": "author", "sentiment": "sentiment_score", "score": "sentiment_score", "collected_at": "observed_at"})
    if "post_id" not in frame:
        raise ValueError("required column missing: post_id (or tweet_id)")
    if "created_at" not in frame:
        raise ValueError("required column missing: created_at (or post_date)")
    if "text" not in frame:
        frame["text"] = ""
    if "author" not in frame:
        frame["author"] = pd.NA
    if "source" not in frame:
        frame["source"] = "unknown"
    if "sentiment_score" not in frame:
        frame["sentiment_score"] = np.nan
    if pd.api.types.is_numeric_dtype(frame["post_id"]):
        frame["post_id"] = frame["post_id"].map(_id_string).astype("string")
    else:
        frame["post_id"] = frame["post_id"].astype("string").str.strip()
        if frame["post_id"].isna().any() or frame["post_id"].eq("").any():
            raise ValueError("post_id contains a missing identifier")
    created, date_only = _utc_series(frame["created_at"], field="created_at", allow_date=True)
    if "time_precision" in frame:
        provided_precision = frame["time_precision"].fillna("")
        invalid_precision = ~provided_precision.isin(["", "date", "timestamp"])
        if invalid_precision.any():
            raise ValueError("time_precision must be 'date' or 'timestamp'")
        if (date_only & provided_precision.eq("timestamp")).any():
            raise ValueError("time_precision='timestamp' contradicts a date-only created_at")
        date_only = date_only | provided_precision.eq("date")
    frame["created_at"] = created
    frame["time_precision"] = np.where(date_only, "date", "timestamp")
    available = pd.Series(created.array, index=frame.index)
    available.loc[date_only] = available.loc[date_only].dt.normalize() + pd.Timedelta("1D")
    for time_column in ("observed_at", "available_at"):
        if time_column not in frame:
            continue
        present = frame[time_column].notna()
        if not present.any():
            continue
        values, _ = _utc_series(frame.loc[present, time_column], field=time_column)
        if (values < created.loc[present]).any():
            raise ValueError(f"{time_column} cannot precede created_at")
        available.loc[present] = pd.concat([available.loc[present], values], axis=1).max(axis=1)
    frame["available_at"] = available
    original_scores = frame["sentiment_score"]
    frame["sentiment_score"] = pd.to_numeric(original_scores, errors="coerce")
    if (original_scores.notna() & frame["sentiment_score"].isna()).any():
        raise ValueError("sentiment_score must be numeric when provided")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["source"] = frame["source"].fillna("unknown").astype(str)
    if frame["sentiment_score"].notna().any() and not np.isfinite(frame.loc[frame["sentiment_score"].notna(), "sentiment_score"].to_numpy(dtype=float)).all():
        raise ValueError("sentiment_score must be finite when provided")
    frame = frame.sort_values(["available_at", "created_at", "post_id"], kind="stable")
    frame = frame.drop_duplicates(["post_id", "ticker"], keep="first")
    return frame[POST_COLUMNS].reset_index(drop=True)


def _dataset_summary(prices: pd.DataFrame, posts: pd.DataFrame, dataset_type: str, **provenance: Any) -> dict[str, Any]:
    return {"dataset_type": dataset_type, "price_rows": len(prices), "post_rows": len(posts), "tickers": sorted(set(prices.ticker) | set(posts.ticker)), "price_date_range": [prices.date.min().date().isoformat(), prices.date.max().date().isoformat()] if len(prices) else None, "post_date_range": [posts.created_at.min().isoformat(), posts.created_at.max().isoformat()] if len(posts) else None, **provenance}


def _write_dataset(prices: pd.DataFrame, posts: pd.DataFrame, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    output = Path(output)
    prices_path, posts_path, metadata_path = output / "prices.csv", output / "posts.csv", output / "dataset.json"
    _atomic_csv(prices, prices_path)
    _atomic_csv(posts, posts_path)
    _atomic_json(metadata, metadata_path)
    return {"output": str(output), "prices_path": str(prices_path), "posts_path": str(posts_path), "metadata_path": str(metadata_path), "price_rows": len(prices), "post_rows": len(posts), "metadata": metadata}


def recover_history(repo: Path, output: Path) -> dict[str, Any]:
    """Recover only the known original price and processed sentiment CSVs."""
    repo = Path(repo)
    price_frames, post_frames = [], []
    for name, ticker in TICKERS.items():
        for relative, destination, frames in ((f"data/stock_prices/{name}_stock_price.csv", "prices", price_frames), (f"data/stock_sentiment/{name}_sentiment.csv", "posts", post_frames)):
            result = subprocess.run(["git", "-C", str(repo), "show", f"{RECOVERY_COMMIT}:{relative}"], capture_output=True, check=False)
            if result.returncode:
                raise ValueError(f"could not recover known CSV {relative} from {RECOVERY_COMMIT}: {result.stderr.decode(errors='replace').strip()}")
            frame = pd.read_csv(io.BytesIO(result.stdout), dtype={"tweet_id": "string"})
            frame["ticker"] = ticker
            frames.append(frame)
    prices = normalize_prices(pd.concat(price_frames, ignore_index=True))
    posts = normalize_posts(pd.concat(post_frames, ignore_index=True))
    metadata = _dataset_summary(prices, posts, "recovered_processed_subset", source_commit=RECOVERY_COMMIT, source_files=[f"data/{kind}/{name}{suffix}" for kind, suffix in (("stock_prices", "_stock_price.csv"), ("stock_sentiment", "_sentiment.csv")) for name in TICKERS], time_assumption="Historical post dates have date-only precision and are available at next UTC midnight.", limitation="Processed non-neutral subset from the original archive; not the full raw Tweet.csv corpus. Historical results are retrospective under an assumed delay.", sentiment_policy="Preserved provided binary sentiment labels; no rescoring.")
    result = _write_dataset(prices, posts, Path(output), metadata)
    return result


def import_original(tweets: Path, mapping: Path, output: Path) -> dict[str, Any]:
    tweets, mapping = Path(tweets), Path(mapping)
    source = pd.read_csv(tweets, dtype={"tweet_id": "string", "post_id": "string", "id": "string"})
    joins = pd.read_csv(mapping, dtype={"tweet_id": "string", "post_id": "string", "id": "string"})
    source = _rename_columns(source, {"tweet_id": "post_id", "id": "post_id", "post_date": "created_at", "date": "created_at", "body": "text", "writer": "author"})
    joins = _rename_columns(joins, {"tweet_id": "post_id", "id": "post_id", "ticker_symbol": "ticker", "symbol": "ticker"})
    if "post_id" not in source or "post_id" not in joins or "ticker" not in joins:
        raise ValueError("tweets and mapping must provide matching tweet_id/post_id columns; mapping also needs ticker_symbol")
    merged = joins.merge(source, on="post_id", how="inner", validate="many_to_one", suffixes=("", "_tweet"))
    if merged.empty:
        raise ValueError("no tweet IDs matched between tweets and mapping")
    posts = normalize_posts(merged)
    result = _write_dataset(pd.DataFrame(columns=PRICE_COLUMNS), posts, Path(output), _dataset_summary(pd.DataFrame(columns=PRICE_COLUMNS), posts, "original_import", tweets_path=str(tweets), mapping_path=str(mapping), limitation="Contains posts only; market prices must be imported separately."))
    return result


def fetch_prices(tickers: list[str], start: str, end: str, output: Path) -> pd.DataFrame:
    """Fetch adjusted prices only when explicitly called by the user."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the 'models' or core project dependencies to use fetch-prices (yfinance is required).") from exc
    tickers = [str(ticker).upper() for ticker in tickers]
    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, actions=False, progress=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no rows; check ticker symbols, date bounds, and network access")
    if isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) == 1:
            raw.columns = raw.columns.get_level_values(0)
        else:
            pieces = []
            for ticker in tickers:
                if ticker in raw.columns.get_level_values(1):
                    piece = raw.xs(ticker, axis=1, level=1).copy()
                elif ticker in raw.columns.get_level_values(0):
                    piece = raw.xs(ticker, axis=1, level=0).copy()
                else:
                    continue
                piece["ticker"] = ticker
                piece.index.name = "date"
                pieces.append(piece.reset_index())
            raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if "ticker" not in raw:
        if len(tickers) != 1:
            raise ValueError("Yahoo Finance response did not identify ticker columns")
        raw = raw.copy()
        raw["ticker"] = tickers[0]
    if "date" not in raw:
        index_name = raw.index.name or "Date"
        raw = raw.reset_index().rename(columns={index_name: "date"})
    prices = normalize_prices(raw)
    _atomic_csv(prices, Path(output) / "prices.csv")
    return prices


def collect_x(query: str, start: str, end: str, ticker: str, output: Path, archive: bool = False, max_pages: int = 10) -> pd.DataFrame:
    """Collect a bounded number of X API v2 search pages using stdlib HTTP."""
    import os
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise ValueError("X_BEARER_TOKEN is required for collect-x")
    if max_pages < 1 or max_pages > 100:
        raise ValueError("max_pages must be between 1 and 100")
    start_at, start_date = _utc_timestamp(start, field="start")
    end_at, end_date = _utc_timestamp(end, field="end")
    if start_date or end_date or start_at >= end_at:
        raise ValueError("start and end must be timezone-aware timestamps with start before end")
    endpoint = "https://api.x.com/2/tweets/search/all" if archive else "https://api.x.com/2/tweets/search/recent"
    rows, next_token = [], None
    for _ in range(max_pages):
        params = {"query": query, "start_time": start_at.isoformat().replace("+00:00", "Z"), "end_time": end_at.isoformat().replace("+00:00", "Z"), "max_results": "100", "tweet.fields": "created_at,author_id"}
        if next_token:
            params["next_token"] = next_token
        request = Request(f"{endpoint}?{urlencode(params)}", headers={"Authorization": f"Bearer {token}", "User-Agent": "sentiment-stock/0.1"})
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
                collected = pd.Timestamp.now(tz="UTC")
                payload = json.loads(body)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"X API request failed ({getattr(exc, 'code', 'network error')}); check API access and date bounds") from exc
        users = {item["id"]: item.get("username") for item in payload.get("includes", {}).get("users", [])}
        for post in payload.get("data", []):
            rows.append({"post_id": post["id"], "ticker": ticker, "created_at": post["created_at"], "observed_at": collected.isoformat(), "text": post.get("text", ""), "author": users.get(post.get("author_id")), "source": "x_api"})
        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break
    posts = normalize_posts(pd.DataFrame(rows, columns=["post_id", "ticker", "created_at", "observed_at", "text", "author", "source"]))
    _atomic_csv(posts, Path(output) / "posts.csv")
    return posts
