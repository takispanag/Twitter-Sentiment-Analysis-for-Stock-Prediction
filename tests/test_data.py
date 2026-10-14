import pandas as pd
import pytest

from sentiment_stock.data import import_original, normalize_posts, normalize_prices


def test_date_only_posts_are_available_after_the_utc_day():
    posts = normalize_posts(pd.DataFrame([{"post_id": 1, "ticker": "AAPL", "created_at": "2020-01-02", "text": "hello"}]))
    assert posts.loc[0, "available_at"] == pd.Timestamp("2020-01-03T00:00:00Z")
    assert posts.loc[0, "time_precision"] == "date"


def test_exported_date_only_posts_keep_their_availability_when_reloaded():
    original = normalize_posts(pd.DataFrame([{"post_id": "1234567890123456789", "ticker": "AAPL", "created_at": "2020-01-02", "text": "hello", "sentiment_score": 1, "source": "archive"}]))
    reloaded = normalize_posts(original)
    assert reloaded.loc[0, "available_at"] == pd.Timestamp("2020-01-03T00:00:00Z")
    assert reloaded.loc[0, "time_precision"] == "date"
    assert reloaded.loc[0, "post_id"] == "1234567890123456789"


def test_observed_at_delays_intraday_post_availability():
    posts = normalize_posts(pd.DataFrame([{"post_id": 1, "ticker": "AAPL", "created_at": "2020-01-02T15:00:00Z", "available_at": "2020-01-02T16:00:00Z", "text": "hello"}]))
    assert posts.loc[0, "available_at"] == pd.Timestamp("2020-01-02T16:00:00Z")


def test_invalid_provided_post_score_is_not_silently_dropped():
    with pytest.raises(ValueError, match="sentiment_score"):
        normalize_posts(pd.DataFrame([{"post_id": 1, "ticker": "AAPL", "created_at": "2020-01-02T15:00:00Z", "text": "hello", "sentiment_score": "not-a-score"}]))


def test_naive_intraday_post_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        normalize_posts(pd.DataFrame([{"post_id": 1, "ticker": "AAPL", "created_at": "2020-01-02 15:00", "text": "hello"}]))


def test_numeric_epoch_post_timestamp_is_accepted_as_seconds():
    posts = normalize_posts(pd.DataFrame([{"post_id": 1, "ticker": "AAPL", "created_at": 1577977200, "text": "hello"}]))
    assert posts.loc[0, "created_at"] == pd.Timestamp("2020-01-02T15:00:00Z")


def test_duplicate_price_sessions_are_rejected():
    row = {"ticker": "AAPL", "date": "2020-11-27", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1}
    with pytest.raises(ValueError, match="duplicate"):
        normalize_prices(pd.DataFrame([row, row]))


def test_prices_use_nyse_early_close_and_reject_holiday():
    row = {"ticker": "AAPL", "date": "2020-11-27", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1}
    prices = normalize_prices(pd.DataFrame([row]))
    assert prices.loc[0, "market_close"] == pd.Timestamp("2020-11-27T18:00:00Z")
    row["date"] = "2020-11-26"
    with pytest.raises(ValueError, match="NYSE session"):
        normalize_prices(pd.DataFrame([row]))


def test_adjusted_close_scales_all_ohlc():
    row = {"ticker": "AAPL", "date": "2020-11-27", "Open": 200, "High": 220, "Low": 90, "Close": 100, "Adj Close": 50, "Volume": 1}
    prices = normalize_prices(pd.DataFrame([row]))
    assert prices.loc[0, ["open", "high", "low", "close"]].tolist() == [100, 110, 45, 50]


def test_original_join_preserves_numeric_ids_and_body(tmp_path):
    tweets = tmp_path / "tweets.csv"
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame([{"tweet_id": "1234567890123456789", "post_date": "2020-01-02T15:00:00Z", "body": "body survives", "writer": "author"}]).to_csv(tweets, index=False)
    pd.DataFrame([{"tweet_id": "1234567890123456789", "ticker_symbol": "AAPL"}, {"tweet_id": "1234567890123456789", "ticker_symbol": "AAPL"}]).to_csv(mapping, index=False)
    result = import_original(tweets, mapping, tmp_path / "out")
    posts = pd.read_csv(result["posts_path"], dtype={"post_id": "string"})
    assert len(posts) == 1
    assert posts.loc[0, "post_id"] == "1234567890123456789"
    assert posts.loc[0, "text"] == "body survives"
