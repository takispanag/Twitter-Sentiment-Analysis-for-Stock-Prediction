"""Causal market and sentiment feature rows aligned to NYSE closes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .data import normalize_posts, normalize_prices

PRICE_FEATURES = ["close_return", "volatility_5", "momentum_5", "momentum_20", "volume_change", "volume_mean_5"]
ATTENTION_FEATURES = ["post_count_log", "unique_authors_log"]
SENTIMENT_FEATURES = ["sentiment_mean", "sentiment_std", "sentiment_positive_fraction", "sentiment_neutral_fraction", "sentiment_negative_fraction", "sentiment_change", "sentiment_mean_5", "sentiment_missing"]


def build_features(prices: pd.DataFrame, posts: pd.DataFrame, target: str = "return", events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per observed close; labels refer only to the immediately following NYSE session."""
    target_map = {"return": "target_return", "next_close": "next_close", "volatility": "target_volatility"}
    if target not in target_map:
        raise ValueError("target must be return, next_close, or volatility")
    prices = normalize_prices(prices) if "market_close" not in prices else prices.copy()
    posts = normalize_posts(posts) if "available_at" not in posts else posts.copy()
    prices["date"] = pd.to_datetime(prices.date).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(prices.date).dt, "tz", None) is not None else pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    posts = posts.sort_values(["ticker", "available_at"], kind="stable").reset_index(drop=True)
    calendar = mcal.get_calendar("NYSE")
    rows: list[dict] = []
    for ticker, market in prices.groupby("ticker", sort=False):
        market = market.reset_index(drop=True)
        ticker_posts = posts.loc[posts.ticker.eq(ticker)].copy()
        closes = pd.to_datetime(market.market_close, utc=True)
        opens = pd.to_datetime(market.market_open, utc=True)
        available_times = pd.to_datetime(ticker_posts.available_at, utc=True).to_numpy(dtype="datetime64[ns]") if len(ticker_posts) else np.array([], dtype="datetime64[ns]")
        days = pd.to_datetime(market.date).dt.normalize()
        sessions = calendar.schedule(start_date=(days.min() - pd.Timedelta(days=10)).date().isoformat(), end_date=days.max().date().isoformat())
        session_closes = pd.Series(pd.to_datetime(sessions.market_close, utc=True).to_numpy(), index=pd.DatetimeIndex(sessions.index).normalize())
        daily_returns = market.close.pct_change()
        trailing_vol = daily_returns.rolling(5, min_periods=2).std()
        session_events = None
        if events is not None and len(events):
            event_rows = events.loc[events.ticker.astype(str).str.upper().eq(str(ticker).upper())] if "ticker" in events else events
            event_dates = pd.to_datetime(event_rows.get("date", event_rows.get("event_date"))).dt.tz_localize(None).dt.normalize()
            session_events = set(event_dates)
        else:
            session_events = set()
        for i, bar in market.iterrows():
            prior_sessions = session_closes.loc[session_closes.index < days.iloc[i]]
            prior_close = prior_sessions.iloc[-1] if len(prior_sessions) else pd.Timestamp("1677-09-22", tz="UTC")
            left = int(np.searchsorted(available_times, np.datetime64(prior_close.tz_localize(None)), side="right")) if len(available_times) else 0
            right = int(np.searchsorted(available_times, np.datetime64(closes.iloc[i].tz_localize(None)), side="right")) if len(available_times) else 0
            available = ticker_posts.iloc[left:right]
            scores = pd.to_numeric(available.get("sentiment_score", pd.Series(dtype=float)), errors="coerce").dropna()
            n = len(available)
            mean = float(scores.mean()) if len(scores) else np.nan
            author = available.get("author", pd.Series(index=available.index, dtype="object"))
            authors = author.dropna().astype(str).str.strip()
            authors = authors[authors.ne("")].nunique()
            pos = float((scores > 0).mean()) if len(scores) else np.nan
            neg = float((scores < 0).mean()) if len(scores) else np.nan
            neu = float((scores == 0).mean()) if len(scores) else np.nan
            next_open = next_close = next_open_at = next_close_at = next_date = actual_return = target_vol = np.nan
            if i + 1 < len(market):
                intended = calendar.schedule(start_date=(days.iloc[i] + pd.Timedelta(days=1)).date().isoformat(), end_date=days.iloc[i + 1].date().isoformat())
                expected_next = pd.Timestamp(intended.index[0]).normalize() if len(intended) else None
                if expected_next is not None and expected_next == days.iloc[i + 1]:
                    nxt = market.iloc[i + 1]
                    next_open, next_close = float(nxt.open), float(nxt.close)
                    next_open_at, next_close_at, next_date = opens.iloc[i + 1], closes.iloc[i + 1], days.iloc[i + 1]
                    actual_return = next_close / next_open - 1
                    target_vol = abs(actual_return)
            prev_cut = pd.to_datetime(bar.market_close, utc=True)
            prior_vol = trailing_vol.iloc[i]
            history = trailing_vol.iloc[:i].dropna()
            if pd.isna(prior_vol) or len(history) < 5:
                regime = "unknown"
            else:
                q = float(history.quantile(.75))
                regime = "high" if prior_vol > q else "low"
            row = {
                "ticker": ticker, "date": days.iloc[i], "feature_date": days.iloc[i], "current_close": float(market.close.iloc[i]),
                "market_open": opens.iloc[i], "market_close": closes.iloc[i],
                "issued_at": closes.iloc[i] + pd.Timedelta(minutes=15), "input_cutoff": closes.iloc[i],
                "target_kind": target, "target_date": next_date, "target_open_at": next_open_at, "target_close_at": next_close_at,
                "next_open": next_open, "next_close": next_close, "target_return": actual_return,
                "actual_return": actual_return, "target_volatility": target_vol,
                "close_return": float(daily_returns.iloc[i]) if pd.notna(daily_returns.iloc[i]) else np.nan,
                "volatility_5": float(prior_vol) if pd.notna(prior_vol) else np.nan,
                "momentum_5": float(market.close.iloc[i] / market.close.iloc[max(0, i - 5)] - 1) if i >= 5 else np.nan,
                "momentum_20": float(market.close.iloc[i] / market.close.iloc[max(0, i - 20)] - 1) if i >= 20 else np.nan,
                "volume_change": float(market.volume.iloc[i] / market.volume.iloc[i - 1] - 1) if i and market.volume.iloc[i - 1] else np.nan,
                "volume_mean_5": float(market.volume.iloc[max(0, i - 4):i + 1].mean()),
                "post_count": n, "post_count_log": float(np.log1p(n)), "unique_authors": authors,
                "unique_authors_log": float(np.log1p(authors)), "sentiment_mean": mean,
                "sentiment_std": float(scores.std(ddof=0)) if len(scores) else np.nan,
                "sentiment_positive_fraction": pos, "sentiment_neutral_fraction": neu, "sentiment_negative_fraction": neg,
                "sentiment_missing": int(not len(scores)), "volatility_regime": regime,
                "event_day": int(days.iloc[i] in session_events),
            }
            row["sentiment_change"] = np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    for _, indices in result.groupby("ticker").groups.items():
        result.loc[indices, "sentiment_change"] = result.loc[indices, "sentiment_mean"].diff().to_numpy()
        result.loc[indices, "sentiment_mean_5"] = result.loc[indices, "sentiment_mean"].rolling(5, min_periods=1).mean().to_numpy()
    result["sentiment_mean_5"] = result["sentiment_mean_5"].fillna(np.nan)
    return result.sort_values(["ticker", "feature_date"], kind="stable").reset_index(drop=True)
