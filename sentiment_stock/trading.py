"""Long/cash backtests for persisted next-session intraday return forecasts."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


_TIME_COLUMNS = ("issued_at", "input_cutoff", "training_cutoff", "target_open_at", "target_close_at")
_REQUIRED = ("run_id", "ticker", "issued_at", "input_cutoff", "training_cutoff", "target_date",
             "target_open_at", "target_close_at", "target_open", "target_close", "target_kind",
             "model", "feature_set", "prediction")


def _finite_nonnegative(name: str, value: float, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _timestamp(value: Any, column: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {column} timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{column} must be a valid timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _validated_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        raise ValueError("forecasts must be a nonempty DataFrame")
    missing = set(_REQUIRED) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast columns missing: {', '.join(sorted(missing))}")
    frame = forecasts.copy()
    for column in _TIME_COLUMNS:
        frame[column] = frame[column].map(lambda value: _timestamp(value, column))
    try:
        dates = pd.to_datetime(frame["target_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("target_date must identify a valid session date") from exc
    if getattr(dates.dt, "tz", None) is not None:
        raise ValueError("target_date must be a date-only session")
    if (dates != dates.dt.normalize()).any():
        raise ValueError("target_date must identify a date-only session")
    frame["target_date"] = dates.dt.normalize()
    frame["ticker"] = frame.ticker.astype("string").str.strip().str.upper()
    if frame.ticker.isna().any() or frame.ticker.eq("").any():
        raise ValueError("forecast ticker values must be present")
    numeric_columns = ["target_open", "target_close", "prediction"]
    if "current_close" in frame:
        numeric_columns.append("current_close")
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values")
        frame[column] = values
    positive_columns = ["target_open", "target_close"] + (["current_close"] if "current_close" in frame else [])
    if (frame[positive_columns] <= 0).any().any():
        raise ValueError("forecast prices must be positive")
    if frame.target_kind.astype(str).ne("return").any():
        raise ValueError("backtest supports only target_kind='return'")
    if (frame.target_close_at <= frame.target_open_at).any():
        raise ValueError("target close must follow target open")
    if (frame.issued_at > frame.target_open_at).any():
        raise ValueError("forecast issue time is after target entry")
    if (frame.training_cutoff >= frame.issued_at).any():
        raise ValueError("training_cutoff must precede issued_at")
    if (frame.input_cutoff > frame.issued_at).any():
        raise ValueError("input_cutoff cannot follow issued_at")
    return frame.sort_values(["target_date", "ticker", "issued_at"], kind="stable").reset_index(drop=True)


def backtest(forecasts: pd.DataFrame, *, model: str = "linear", feature_set: str = "price_sentiment",
             threshold: float = 0.0, cost_bps: float = 5.0, slippage_bps: float = 5.0,
             initial_cash: float = 10000.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply one long/cash decision per ticker and session.

    Benchmark allocations buy at each ticker's first eligible forecast open,
    include each available subsequent close (and therefore overnight gaps),
    and remain marked to market at the last close without liquidation fees.
    """
    threshold = _finite_nonnegative("threshold", threshold)
    cost_bps = _finite_nonnegative("cost_bps", cost_bps)
    slippage_bps = _finite_nonnegative("slippage_bps", slippage_bps)
    initial_cash = _finite_nonnegative("initial_cash", initial_cash, positive=True)
    all_rows = _validated_forecasts(forecasts)
    rows = all_rows.loc[all_rows.model.astype(str).eq(str(model)) & all_rows.feature_set.astype(str).eq(str(feature_set))].copy()
    if rows.empty:
        raise ValueError(f"no forecasts found for model={model!r} and feature_set={feature_set!r}")
    if rows.duplicated(["ticker", "target_date"]).any():
        raise ValueError("duplicate ticker/session forecasts for selected strategy")
    tickers = sorted(rows.ticker.unique().tolist())
    allocation = initial_cash / len(tickers)
    fee_rate, slip_rate = cost_bps / 10000.0, slippage_bps / 10000.0
    net_values = {ticker: allocation for ticker in tickers}
    gross_values = {ticker: allocation for ticker in tickers}
    benchmark_shares: dict[str, float] = {}
    benchmark_last: dict[str, float] = {}
    by_day = {(ticker, date): group.iloc[0] for (ticker, date), group in rows.groupby(["ticker", "target_date"], sort=False)}
    sessions = sorted(pd.Timestamp(value) for value in rows.target_date.unique())
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    for session in sessions:
        exposure = traded = 0.0
        for ticker in tickers:
            row = by_day.get((ticker, session))
            if row is None:
                continue
            open_price, close_price = float(row.target_open), float(row.target_close)
            if ticker not in benchmark_shares:
                benchmark_entry = open_price * (1 + slip_rate)
                benchmark_shares[ticker] = allocation / (benchmark_entry * (1 + fee_rate))
            benchmark_last[ticker] = benchmark_shares[ticker] * close_price
            current_net, current_gross = net_values[ticker], gross_values[ticker]
            if float(row.prediction) > threshold:
                entry = open_price * (1 + slip_rate)
                exit_ = close_price * (1 - slip_rate)
                shares = current_net / (entry * (1 + fee_rate))
                entry_notional, exit_notional = shares * entry, shares * exit_
                entry_fee, exit_fee = entry_notional * fee_rate, exit_notional * fee_rate
                net_end = exit_notional - exit_fee
                gross_shares = current_gross / open_price
                gross_end = gross_shares * close_price
                net_values[ticker], gross_values[ticker] = net_end, gross_end
                exposure += allocation / initial_cash
                traded += entry_notional + exit_notional
                trade = {column: row[column] for column in rows.columns}
                trade.update({"session": session, "shares": shares, "entry_price": entry, "exit_price": exit_,
                              "entry_fee": entry_fee, "exit_fee": exit_fee,
                              "entry_notional": entry_notional, "exit_notional": exit_notional,
                              "gross_return": close_price / open_price - 1,
                              "net_return": net_end / current_net - 1,
                              "gross_equity_after": gross_end, "net_equity_after": net_end})
                trades.append(trade)
        net_equity = sum(net_values.values())
        gross_equity = sum(gross_values.values())
        benchmark_equity = sum(benchmark_last.get(ticker, allocation) for ticker in tickers)
        equity_rows.append({"session": session, "gross_equity": gross_equity, "net_equity": net_equity,
                            "benchmark_equity": benchmark_equity, "cash": net_equity,
                            "exposure": exposure, "traded_notional": traded})
    equity = pd.DataFrame(equity_rows)
    for column in ("gross_equity", "net_equity", "benchmark_equity"):
        equity[f"{column}_return"] = equity[column].pct_change().fillna(equity[column].iloc[0] / initial_cash - 1)
        equity[f"{column}_drawdown"] = equity[column] / equity[column].cummax() - 1
    total_return = float(equity.net_equity.iloc[-1] / initial_cash - 1)
    gross_return = float(equity.gross_equity.iloc[-1] / initial_cash - 1)
    benchmark_return = float(equity.benchmark_equity.iloc[-1] / initial_cash - 1)
    summary = {"model": model, "feature_set": feature_set, "threshold": threshold, "cost_bps_per_side": cost_bps,
               "slippage_bps_per_side": slippage_bps, "initial_cash": initial_cash,
               "tickers": tickers, "sessions": len(equity), "trades": len(trades), "net_return": total_return,
               "gross_return": gross_return, "benchmark_return": benchmark_return,
               "max_drawdown": float(equity.net_equity_drawdown.min()),
               "benchmark_max_drawdown": float(equity.benchmark_equity_drawdown.min()),
               "mean_exposure": float(equity.exposure.mean()), "total_traded_notional": float(equity.traded_notional.sum()),
               "benchmark_treatment": "Equal allocations buy at each ticker's first eligible open; entry fee and slippage apply; marked to market through available subsequent closes, with no final liquidation fee."}
    return equity, pd.DataFrame(trades), summary


def cost_sensitivity(forecasts: pd.DataFrame, **strategy: Any) -> pd.DataFrame:
    """Re-run a fixed strategy across labelled per-side fee assumptions."""
    slippage = strategy.pop("slippage_bps", 5.0)
    strategy.pop("cost_bps", None)
    results = []
    for cost in (0.0, 5.0, 10.0, 25.0):
        _, trades, summary = backtest(forecasts, cost_bps=cost, slippage_bps=slippage, **strategy)
        results.append({"cost_bps": cost, "cost_label": f"{cost:g} bps per side", "slippage_bps": slippage,
                        "net_return": summary["net_return"], "gross_return": summary["gross_return"],
                        "benchmark_return": summary["benchmark_return"], "trades": len(trades)})
    return pd.DataFrame(results)
