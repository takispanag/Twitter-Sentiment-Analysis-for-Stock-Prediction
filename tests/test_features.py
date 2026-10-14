import numpy as np
import pandas as pd
import pytest

from sentiment_stock.features import build_features


def market_inputs():
    prices = pd.DataFrame({"ticker": ["XYZ"] * 4, "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]), "open": [90, 100, 101, 103], "high": [101, 111, 104, 106], "low": [89, 99, 100, 102], "close": [100, 110, 102, 105], "volume": [10, 20, 30, 40], "market_open": pd.to_datetime(["2024-01-02 14:30Z", "2024-01-03 14:30Z", "2024-01-04 14:30Z", "2024-01-05 14:30Z"]), "market_close": pd.to_datetime(["2024-01-02 21:00Z", "2024-01-03 21:00Z", "2024-01-04 21:00Z", "2024-01-05 21:00Z"])})
    posts = pd.DataFrame({"post_id": ["a", "b"], "ticker": ["XYZ", "XYZ"], "created_at": pd.to_datetime(["2024-01-02 20:00Z", "2024-01-03 21:01Z"], utc=True), "available_at": pd.to_datetime(["2024-01-02 20:00Z", "2024-01-03 21:01Z"], utc=True), "sentiment_score": [1.0, -1.0], "author": ["alice", None], "time_precision": ["timestamp", "timestamp"]})
    return prices, posts


def test_targets_use_next_session_and_empty_coverage_is_kept():
    prices, posts = market_inputs()
    got = build_features(prices, posts)
    row = got.loc[got.feature_date.eq(pd.Timestamp("2024-01-02"))].iloc[0]
    assert row.target_return == pytest.approx(.10)
    assert row.next_close == 110
    assert row.post_count == 1 and row.unique_authors == 1
    no_coverage = got.loc[got.feature_date.eq(pd.Timestamp("2024-01-03"))].iloc[0]
    delayed = got.loc[got.feature_date.eq(pd.Timestamp("2024-01-04"))].iloc[0]
    assert no_coverage.post_count == 0 and no_coverage.sentiment_missing == 1
    assert delayed.post_count == 1 and delayed.unique_authors == 0


def test_posts_after_close_cannot_change_the_closed_feature_row():
    prices, posts = market_inputs()
    baseline = build_features(prices, posts)
    future = pd.concat([posts, pd.DataFrame({"post_id": ["late"], "ticker": ["XYZ"], "created_at": [pd.Timestamp("2024-01-03 22:00Z")], "available_at": [pd.Timestamp("2024-01-03 22:00Z")], "sentiment_score": [1.0], "author": ["bob"], "time_precision": ["timestamp"]})], ignore_index=True)
    changed = build_features(prices, future)
    cols = ["post_count", "sentiment_mean", "sentiment_missing"]
    pd.testing.assert_series_equal(baseline.loc[1, cols], changed.loc[1, cols], check_names=False)


def test_missing_calendar_session_does_not_make_later_bar_tomorrow():
    prices, posts = market_inputs()
    prices = prices.drop(index=1).reset_index(drop=True)
    got = build_features(prices, posts)
    assert pd.isna(got.iloc[0].next_open) and pd.isna(got.iloc[0].target_return)
