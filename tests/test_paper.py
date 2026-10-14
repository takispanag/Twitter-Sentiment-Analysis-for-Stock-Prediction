from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from sentiment_stock.paper import read_paper, record_forecasts, settle_forecasts


NOW = pd.Timestamp("2026-01-05T21:30:00Z")


def forward(ticker="AAA", target="2026-01-06", *, prediction=0.1, run="r1"):
    return {
        "run_id": run, "ticker": ticker, "feature_date": "2026-01-05",
        "issued_at": "2026-01-05T21:15:00Z", "input_cutoff": "2026-01-05T21:00:00Z",
        "training_cutoff": "2026-01-04T21:00:00Z", "target_date": target,
        "target_open_at": f"{target}T14:30:00Z", "target_close_at": f"{target}T21:00:00Z",
        "current_close": 100.0, "target_kind": "return", "model": "linear",
        "feature_set": "price_sentiment", "fold": "forward", "prediction": prediction,
    }


def prices(*, ticker="AAA", day="2026-01-06", open_=100.0, close=110.0):
    return pd.DataFrame([{"ticker": ticker, "date": day, "open": open_, "close": close,
                          "market_open": f"{day}T14:30:00Z", "market_close": f"{day}T21:00:00Z"}])


def test_read_missing_database_does_not_create_it(tmp_path):
    db = tmp_path / "paper.sqlite"
    assert read_paper(db).empty
    assert not db.exists()


def test_forward_record_is_idempotent_and_changed_identity_is_rejected(tmp_path):
    db = tmp_path / "paper.sqlite"
    batch = pd.DataFrame([forward()])
    assert record_forecasts(db, batch, recorded_at=NOW) == 1
    assert record_forecasts(db, batch, recorded_at=NOW) == 0
    changed = batch.copy()
    changed.loc[0, "prediction"] = 0.2
    with pytest.raises(ValueError, match="changed"):
        record_forecasts(db, changed, recorded_at=NOW)
    stored = read_paper(db)
    assert len(stored) == 1
    assert stored.iloc[0].prediction == pytest.approx(0.1)
    assert pd.Timestamp(stored.iloc[0].recorded_at) == NOW


def test_record_validation_rejects_past_target_actuals_and_temporal_violations(tmp_path):
    db = tmp_path / "paper.sqlite"
    bad = [forward(target="2026-01-02"),
           {**forward(), "actual": 0.1},
           {**forward(), "target_open": 100},
           {**forward(), "issued_at": "2026-01-05T14:01:00Z"},
           {**forward(), "training_cutoff": "2026-01-05T21:15:00Z"},
           {**forward(), "input_cutoff": "2026-01-05T14:01:00Z"}]
    for row in bad:
        with pytest.raises(ValueError):
            record_forecasts(db, pd.DataFrame([row]), recorded_at=NOW)
    assert read_paper(db).empty


def test_entire_batch_validates_before_any_insert(tmp_path):
    db = tmp_path / "paper.sqlite"
    batch = pd.DataFrame([forward(), {**forward("BBB"), "target_close_at": "invalid"}])
    with pytest.raises(ValueError):
        record_forecasts(db, batch, recorded_at=NOW)
    assert read_paper(db).empty


def test_record_rejects_feature_and_target_time_inconsistency(tmp_path):
    db = tmp_path / "paper.sqlite"
    rows = [
        {**forward(), "feature_date": "2026-01-07"},
        {**forward(), "issued_at": "2026-01-04T23:00:00Z", "input_cutoff": "2026-01-04T23:00:00Z"},
        {**forward(), "target_open_at": "2026-01-07T14:30:00Z", "target_close_at": "2026-01-07T21:00:00Z"},
    ]
    for row in rows:
        with pytest.raises(ValueError):
            record_forecasts(db, pd.DataFrame([row]), recorded_at=NOW)
    assert read_paper(db).empty


def test_record_rejects_unfinished_feature_session(tmp_path):
    db = tmp_path / "paper.sqlite"
    row = {**forward(), "issued_at": "2026-01-05T13:00:00Z", "input_cutoff": "2026-01-05T12:45:00Z"}
    with pytest.raises(ValueError, match="close"):
        record_forecasts(db, pd.DataFrame([row]), recorded_at="2026-01-05T14:00:00Z")
    assert read_paper(db).empty


def test_record_rejects_wrong_market_hours_on_a_real_session(tmp_path):
    db = tmp_path / "paper.sqlite"
    row = {**forward(), "target_open_at": "2026-01-06T10:00:00Z", "target_close_at": "2026-01-06T20:00:00Z"}
    with pytest.raises(ValueError, match="NYSE"):
        record_forecasts(db, pd.DataFrame([row]), recorded_at=NOW)


@pytest.mark.parametrize("target,feature,cutoff,issue,recorded,open_at,close_at", [
    ("2025-07-03", "2025-07-02", "2025-07-02T20:00:00Z", "2025-07-02T20:15:00Z", "2025-07-02T20:30:00Z", "2025-07-03T13:30:00Z", "2025-07-03T17:00:00Z"),
    ("2025-11-28", "2025-11-26", "2025-11-26T21:00:00Z", "2025-11-26T21:15:00Z", "2025-11-26T21:30:00Z", "2025-11-28T14:30:00Z", "2025-11-28T18:00:00Z"),
])
def test_record_accepts_nyse_early_close_boundaries_across_dst(tmp_path, target, feature, cutoff, issue, recorded, open_at, close_at):
    db = tmp_path / f"{target}.sqlite"
    row = {**forward(target=target), "feature_date": feature, "input_cutoff": cutoff, "issued_at": issue,
           "training_cutoff": "2025-07-01T20:00:00Z" if target == "2025-07-03" else "2025-11-25T21:00:00Z",
           "target_open_at": open_at, "target_close_at": close_at}
    assert record_forecasts(db, pd.DataFrame([row]), recorded_at=recorded) == 1


def test_recorded_strategy_is_immutable_through_settlement_and_reopen(tmp_path):
    db = tmp_path / "paper.sqlite"
    record_forecasts(db, pd.DataFrame([forward()]), recorded_at=NOW, threshold=0.05,
                     cost_bps=10, slippage_bps=20)
    assert settle_forecasts(db, prices(), as_of="2026-01-06T20:59:59Z") == 0
    assert read_paper(db).iloc[0].status == "pending"
    assert settle_forecasts(db, prices(), as_of="2026-01-06T21:00:00Z") == 1
    assert settle_forecasts(db, prices(), as_of="2026-01-06T22:00:00Z") == 0
    stored = read_paper(db)
    assert stored.iloc[0].status == "settled"
    assert stored.iloc[0].threshold == pytest.approx(0.05)
    # Positive prediction exceeds the threshold; entry and exit fills retain recorded slippage.
    assert stored.iloc[0].entry_price == pytest.approx(100.2)
    assert stored.iloc[0].exit_price == pytest.approx(109.78)
    assert stored.iloc[0].net_return == pytest.approx((109.78 * (1 - .001)) / (100.2 * (1 + .001)) - 1)
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from forecasts").fetchone()[0] == 1


def test_settlement_rejects_mismatched_session_and_rewrite_of_settled_prices(tmp_path):
    db = tmp_path / "paper.sqlite"
    record_forecasts(db, pd.DataFrame([forward()]), recorded_at=NOW)
    with pytest.raises(ValueError, match="matching"):
        settle_forecasts(db, prices(ticker="BBB"), as_of="2026-01-06T22:00:00Z")
    settle_forecasts(db, prices(), as_of="2026-01-06T22:00:00Z")
    with pytest.raises(ValueError, match="mismatch"):
        settle_forecasts(db, prices(close=111), as_of="2026-01-07T22:00:00Z")


def test_settlement_rejects_duplicate_prices_after_ticker_normalization(tmp_path):
    db = tmp_path / "paper.sqlite"
    record_forecasts(db, pd.DataFrame([forward()]), recorded_at=NOW)
    duplicate = pd.concat([prices(), prices(ticker="aaa", close=120)], ignore_index=True)
    duplicate.loc[1, "date"] = pd.Timestamp("2026-01-06")
    with pytest.raises(ValueError, match="duplicate"):
        settle_forecasts(db, duplicate, as_of="2026-01-06T22:00:00Z")
    assert read_paper(db).iloc[0].status == "pending"


def test_identity_allows_distinct_strategy_experiments(tmp_path):
    db = tmp_path / "paper.sqlite"
    rows = pd.DataFrame([forward(run="r1"), forward(run="r2")])
    assert record_forecasts(db, rows, recorded_at=NOW) == 2
    assert len(read_paper(db)) == 2
