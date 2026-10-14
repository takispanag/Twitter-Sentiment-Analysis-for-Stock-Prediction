"""Deterministic synthetic inputs for offline demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .data import _dataset_summary, _write_dataset, normalize_posts, normalize_prices


def make_demo(output: Path, sessions: int = 260, seed: int = 11) -> dict:
    if sessions < 2:
        raise ValueError("sessions must be at least 2")
    tickers = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]
    calendar = mcal.get_calendar("NYSE")
    days = max(30, int(sessions * 1.5))
    start_date = (pd.Timestamp("2025-12-31") - pd.Timedelta(days=days)).date().isoformat()
    schedule = calendar.schedule(start_date=start_date, end_date="2025-12-31")
    dates = schedule.index[-sessions:]
    rng = np.random.default_rng(seed)
    price_rows, post_rows = [], []
    for ticker in tickers:
        prior = float(rng.uniform(40, 250))
        for index, session in enumerate(dates):
            gap = float(np.clip(rng.normal(0, 0.006), -0.025, 0.025))
            intraday = float(np.clip(rng.normal(0, 0.012), -0.045, 0.045))
            open_price = prior * (1 + gap)
            close_price = open_price * (1 + intraday)
            high = max(open_price, close_price) * (1 + float(rng.uniform(0.001, 0.012)))
            low = min(open_price, close_price) * (1 - float(rng.uniform(0.001, 0.012)))
            price_rows.append({"ticker": ticker, "date": session.date().isoformat(), "open": open_price, "high": high, "low": low, "close": close_price, "volume": int(rng.integers(1_000_000, 20_000_000))})
            created = pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta("14h30min") + pd.Timedelta(int(rng.integers(0, 360)), unit="m")
            post_rows.append({"post_id": f"demo-{seed}-{ticker}-{index}", "ticker": ticker, "created_at": created.isoformat(), "text": f"Synthetic market discussion for {ticker}.", "author": "synthetic", "sentiment_score": float(rng.uniform(-1, 1)), "source": "synthetic_demo"})
            prior = close_price
    prices = normalize_prices(pd.DataFrame(price_rows))
    posts = normalize_posts(pd.DataFrame(post_rows))
    metadata = _dataset_summary(prices, posts, "synthetic", seed=seed, sessions=sessions, limitation="Synthetic independent noise; not evidence of predictive skill or profitability.", sentiment_generation="Scores are independent random draws and are not generated from or aligned to future price labels.")
    result = _write_dataset(prices, posts, Path(output), metadata)
    return result
