from __future__ import annotations

import pandas as pd
import pytest

from sentiment_stock.trading import backtest, cost_sensitivity


def forecast(ticker="AAA", day="2026-01-05", *, prediction=0.1, open_=100.0, close=110.0,
             run="r1", model="linear", feature_set="price_sentiment", issued=None,
             target_kind="return", training=None):
    target = pd.Timestamp(day)
    return {
        "run_id": run, "ticker": ticker, "feature_date": target - pd.Timedelta(days=1),
        "issued_at": pd.Timestamp(issued or "2026-01-02T21:15:00Z"),
        "input_cutoff": pd.Timestamp("2026-01-02T21:00:00Z"),
        "training_cutoff": pd.Timestamp(training or "2026-01-01T21:00:00Z"),
        "target_date": target, "target_open_at": pd.Timestamp(f"{day}T14:30:00Z"),
        "target_close_at": pd.Timestamp(f"{day}T21:00:00Z"), "target_open": open_,
        "target_close": close, "current_close": 99.0, "target_kind": target_kind,
        "model": model, "feature_set": feature_set, "fold": 1,
        "prediction": prediction, "actual": close / open_ - 1,
    }


def run(rows, **kwargs):
    return backtest(pd.DataFrame(rows), initial_cash=1000.0, cost_bps=0.0, slippage_bps=0.0, **kwargs)


def test_long_prediction_earns_open_to_close_return():
    equity, trades, summary = run([forecast()])
    assert equity.iloc[-1].net_equity == pytest.approx(1100.0)
    assert equity.iloc[-1].gross_equity == pytest.approx(1100.0)
    assert trades.iloc[0].entry_price == 100.0
    assert trades.iloc[0].exit_price == 110.0
    assert summary["net_return"] == pytest.approx(0.1)


def test_negative_or_zero_prediction_stays_in_cash():
    for prediction in (-0.1, 0.0):
        equity, trades, _ = run([forecast(prediction=prediction)])
        assert equity.iloc[-1].net_equity == pytest.approx(1000.0)
        assert trades.empty


def test_per_side_fee_uses_actual_entry_and_exit_notionals():
    equity, trades, _ = backtest(pd.DataFrame([forecast()]), initial_cash=1000,
                                  cost_bps=10, slippage_bps=0)
    # Shares = 1000 / (100 * (1 + 0.001)); sell value = shares * 110.
    shares = 1000 / (100 * 1.001)
    expected = shares * 110 * (1 - 0.001)
    assert trades.iloc[0].entry_fee == pytest.approx(1000 - shares * 100)
    assert trades.iloc[0].exit_fee == pytest.approx(shares * 110 * 0.001)
    assert equity.iloc[-1].net_equity == pytest.approx(expected)


def test_slippage_changes_entry_and_exit_fill_prices():
    _, trades, _ = backtest(pd.DataFrame([forecast()]), initial_cash=1000,
                            cost_bps=0, slippage_bps=100)
    assert trades.iloc[0].entry_price == pytest.approx(101.0)
    assert trades.iloc[0].exit_price == pytest.approx(108.9)


def test_equal_ticker_allocations_prevent_full_capital_profit_per_ticker():
    rows = [forecast("AAA", prediction=0.1), forecast("BBB", prediction=0.1, close=90)]
    equity, _, _ = run(rows)
    assert equity.iloc[-1].net_equity == pytest.approx(1000.0)


def test_missing_forecast_session_stays_cash_and_buy_hold_includes_overnight_gap():
    rows = [forecast(day="2026-01-05", prediction=0.1, open_=100, close=100),
            forecast(day="2026-01-07", prediction=-0.1, open_=120, close=132)]
    equity, _, _ = run(rows)
    assert list(equity.session.astype(str)) == ["2026-01-05", "2026-01-07"]
    assert equity.iloc[-1].net_equity == pytest.approx(1000.0)
    assert equity.iloc[-1].benchmark_equity == pytest.approx(1320.0)


@pytest.mark.parametrize("changes", [
    {"target_kind": "next_close"},
    {"issued_at": "2026-01-05T14:31:00Z"},
    {"training_cutoff": "2026-01-05T14:30:00Z"},
    {"target_open": float("nan")},
    {"current_close": float("nan")},
])
def test_rejects_ineligible_or_malformed_forecasts(changes):
    row = forecast()
    row.update(changes)
    with pytest.raises(ValueError):
        backtest(pd.DataFrame([row]))


def test_rejects_duplicate_ticker_session_and_invalid_strategy_values():
    with pytest.raises(ValueError, match="duplicate"):
        run([forecast(), forecast(run="other")])
    for kwargs in ({"initial_cash": -1}, {"cost_bps": -1}, {"slippage_bps": float("inf")}):
        with pytest.raises(ValueError):
            backtest(pd.DataFrame([forecast()]), **kwargs)


def test_cost_sensitivity_labels_per_side_fee_scenarios():
    results = cost_sensitivity(pd.DataFrame([forecast()]), initial_cash=1000, slippage_bps=0, cost_bps=5)
    assert list(results.cost_bps) == [0, 5, 10, 25]
    assert results.net_return.is_monotonic_decreasing
