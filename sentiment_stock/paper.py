"""Append-only SQLite ledger for forward forecasts and later paper settlement."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


_IDENTITY = ("run_id", "model", "feature_set", "ticker", "target_kind", "target_date")
_REQUIRED = (*_IDENTITY, "feature_date", "issued_at", "input_cutoff", "training_cutoff", "target_open_at", "target_close_at",
             "prediction")
_FORBIDDEN = {"actual", "actual_return", "target_open", "target_close", "next_open", "next_close"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
  run_id TEXT NOT NULL, model TEXT NOT NULL, feature_set TEXT NOT NULL, ticker TEXT NOT NULL,
  target_kind TEXT NOT NULL, target_date TEXT NOT NULL, original_json TEXT NOT NULL,
  recorded_at TEXT NOT NULL, threshold REAL NOT NULL, cost_bps REAL NOT NULL, slippage_bps REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', settled_open REAL, settled_close REAL,
  entry_price REAL, exit_price REAL, entry_fee REAL, exit_fee REAL, gross_return REAL, net_return REAL,
  settled_at TEXT,
  PRIMARY KEY (run_id, model, feature_set, ticker, target_kind, target_date)
)
"""


def _stamp(value: Any, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name} timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def _number(value: Any, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    return value


def _prepared(forecasts: pd.DataFrame, recorded_at: pd.Timestamp, threshold: float,
              cost_bps: float, slippage_bps: float) -> list[dict[str, Any]]:
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        raise ValueError("forecasts must be a nonempty DataFrame")
    forbidden = [column for column in forecasts.columns if column.casefold() in _FORBIDDEN or
                 column.casefold().startswith("actual_") or
                 (column.casefold().startswith("target_") and column.casefold() not in {"target_date", "target_kind", "target_open_at", "target_close_at"})]
    if forbidden:
        raise ValueError(f"forward forecasts cannot contain realized values: {', '.join(forbidden)}")
    missing = set(_REQUIRED) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast columns missing: {', '.join(sorted(missing))}")
    result = []
    seen = set()
    for raw in forecasts.to_dict("records"):
        row = {str(key): _json_value(value) for key, value in raw.items()}
        if any(row.get(key) is None or str(row.get(key)).strip() == "" for key in _IDENTITY):
            raise ValueError("forecast identity fields must be present")
        row["ticker"] = str(row["ticker"]).strip().upper()
        if row["target_kind"] != "return":
            raise ValueError("paper ledger supports only target_kind='return'")
        try:
            target_date = pd.Timestamp(row["target_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError("target_date must identify a target session") from exc
        if pd.isna(target_date) or (target_date.tzinfo is not None) or target_date != target_date.normalize():
            raise ValueError("target_date must be a date-only session")
        row["target_date"] = target_date.date().isoformat()
        try:
            feature_date = pd.Timestamp(row["feature_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_date must identify a feature session") from exc
        if pd.isna(feature_date) or feature_date.tzinfo is not None or feature_date != feature_date.normalize():
            raise ValueError("feature_date must be a date-only session")
        if feature_date >= target_date:
            raise ValueError("feature_date must precede target_date")
        row["feature_date"] = feature_date.date().isoformat()
        stamps = {name: _stamp(row[name], name) for name in ("issued_at", "input_cutoff", "training_cutoff", "target_open_at", "target_close_at")}
        if stamps["issued_at"] > recorded_at or recorded_at >= stamps["target_open_at"]:
            raise ValueError("recorded_at must satisfy issued_at <= recorded_at < target_open_at")
        if stamps["training_cutoff"] >= stamps["issued_at"]:
            raise ValueError("training_cutoff must precede issued_at")
        if stamps["input_cutoff"] >= stamps["issued_at"]:
            raise ValueError("issued_at must follow input_cutoff")
        if stamps["target_close_at"] <= stamps["target_open_at"]:
            raise ValueError("target_close_at must follow target_open_at")
        if feature_date > stamps["issued_at"].tz_localize(None).normalize():
            raise ValueError("issued_at precedes the feature session")
        if stamps["target_open_at"].date().isoformat() != row["target_date"] or stamps["target_close_at"].date().isoformat() != row["target_date"]:
            raise ValueError("target execution timestamps must match target_date")
        row.update({name: stamp.isoformat() for name, stamp in stamps.items()})
        row["prediction"] = _number(row["prediction"], "prediction")
        key = tuple(str(row[name]) for name in _IDENTITY)
        if key in seen:
            raise ValueError("duplicate forecast identity in batch")
        seen.add(key)
        input_json = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result.append({"key": key, "row": row, "input_json": input_json})
    return result


def _connect(db: Path) -> sqlite3.Connection:
    path = Path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    return connection


def record_forecasts(db: Path, forecasts: pd.DataFrame, *, recorded_at: pd.Timestamp | None = None,
                     threshold: float = 0.0, cost_bps: float = 5.0, slippage_bps: float = 5.0) -> int:
    clock = _stamp(recorded_at if recorded_at is not None else pd.Timestamp.now(tz="UTC"), "recorded_at")
    threshold = _number(threshold, "threshold", nonnegative=True)
    cost_bps = _number(cost_bps, "cost_bps", nonnegative=True)
    slippage_bps = _number(slippage_bps, "slippage_bps", nonnegative=True)
    prepared = _prepared(forecasts, clock, threshold, cost_bps, slippage_bps)
    connection = _connect(db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(_SCHEMA)
        # Check every retry before inserting any row, so a changed member cannot partially append a batch.
        for item in prepared:
            existing = connection.execute("SELECT original_json, threshold, cost_bps, slippage_bps FROM forecasts WHERE run_id=? AND model=? AND feature_set=? AND ticker=? AND target_kind=? AND target_date=?", item["key"]).fetchone()
            if existing is not None and (existing["original_json"] != item["input_json"] or
                    existing["threshold"] != threshold or existing["cost_bps"] != cost_bps or
                    existing["slippage_bps"] != slippage_bps):
                raise ValueError("forecast identity already exists with changed input or strategy")
        inserted = 0
        for item in prepared:
            cursor = connection.execute("INSERT OR IGNORE INTO forecasts (run_id,model,feature_set,ticker,target_kind,target_date,original_json,recorded_at,threshold,cost_bps,slippage_bps) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (*item["key"], item["input_json"], clock.isoformat(), threshold, cost_bps, slippage_bps))
            inserted += cursor.rowcount
        connection.commit()
        return inserted
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _price_rows(prices: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    required = {"ticker", "date", "open", "close", "market_open", "market_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"canonical prices missing columns: {', '.join(sorted(missing))}")
    result = {}
    for raw in prices.to_dict("records"):
        ticker = str(raw["ticker"]).strip().upper()
        date = pd.Timestamp(raw["date"])
        if date.tzinfo is not None or date != date.normalize():
            raise ValueError("price date must be a date-only session")
        open_, close = _number(raw["open"], "open", positive=True), _number(raw["close"], "close", positive=True)
        market_open, market_close = _stamp(raw["market_open"], "market_open"), _stamp(raw["market_close"], "market_close")
        if market_close <= market_open:
            raise ValueError("market_close must follow market_open")
        key = (ticker, date.date().isoformat())
        if key in result:
            raise ValueError("duplicate ticker/session prices after normalization")
        result[key] = {"open": open_, "close": close,
              "market_open": market_open.isoformat(), "market_close": market_close.isoformat()}
    return result


def settle_forecasts(db: Path, prices: pd.DataFrame, *, as_of: pd.Timestamp | None = None) -> int:
    clock = _stamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC"), "as_of")
    price_map = _price_rows(prices)
    path = Path(db)
    if not path.exists():
        return 0
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute("SELECT * FROM forecasts ORDER BY target_date,ticker,run_id,model,feature_set").fetchall()
        changes = []
        for stored in rows:
            item = dict(stored)
            original = json.loads(item["original_json"])
            key = (item["ticker"], item["target_date"])
            supplied = price_map.get(key)
            if supplied is None:
                if item["status"] == "settled":
                    continue
                if _stamp(original["target_close_at"], "target_close_at") <= clock:
                    raise ValueError(f"missing matching canonical prices for {item['ticker']} {item['target_date']}")
                continue
            if supplied["market_open"] != original["target_open_at"] or supplied["market_close"] != original["target_close_at"]:
                raise ValueError("canonical prices do not match forecast target session times")
            if item["status"] == "settled":
                if supplied["open"] != item["settled_open"] or supplied["close"] != item["settled_close"]:
                    raise ValueError("settled price mismatch; ledger outcomes are immutable")
                continue
            if _stamp(original["target_close_at"], "target_close_at") > clock:
                continue
            prediction, threshold = float(original["prediction"]), item["threshold"]
            fee, slip = item["cost_bps"] / 10000.0, item["slippage_bps"] / 10000.0
            if prediction > threshold:
                entry = supplied["open"] * (1 + slip)
                exit_ = supplied["close"] * (1 - slip)
                entry_fee, exit_fee = entry * fee, exit_ * fee
                gross_return, net_return = supplied["close"] / supplied["open"] - 1, (exit_ * (1 - fee)) / (entry * (1 + fee)) - 1
            else:
                entry = exit_ = entry_fee = exit_fee = 0.0
                gross_return = net_return = 0.0
            changes.append((supplied["open"], supplied["close"], entry, exit_, entry_fee, exit_fee,
                            gross_return, net_return, clock.isoformat(), *tuple(item[name] for name in _IDENTITY)))
        for values in changes:
            connection.execute("UPDATE forecasts SET status='settled',settled_open=?,settled_close=?,entry_price=?,exit_price=?,entry_fee=?,exit_fee=?,gross_return=?,net_return=?,settled_at=? WHERE run_id=? AND model=? AND feature_set=? AND ticker=? AND target_kind=? AND target_date=? AND status='pending'", values)
        connection.commit()
        return len(changes)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def read_paper(db: Path) -> pd.DataFrame:
    """Read pending and settled records without creating a missing database."""
    path = Path(db)
    if not path.exists():
        return pd.DataFrame()
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        try:
            records = connection.execute("SELECT * FROM forecasts ORDER BY target_date,ticker,run_id,model,feature_set").fetchall()
            columns = [item[0] for item in connection.execute("SELECT * FROM forecasts LIMIT 0").description]
        except sqlite3.OperationalError:
            return pd.DataFrame()
        rows = []
        for record in records:
            item = dict(zip(columns, record))
            rows.append({**json.loads(item.pop("original_json")), **item})
        return pd.DataFrame(rows)
    finally:
        connection.close()
