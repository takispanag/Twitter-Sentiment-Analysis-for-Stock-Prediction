import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sentiment_stock.sequence import sequence_windows
from sentiment_stock.research import run_research, forecast_latest, _comparisons
from sentiment_stock.demo import make_demo


def research_frame(rows=36):
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame({"ticker": "XYZ", "feature_date": dates, "issued_at": dates.tz_localize("UTC") + pd.Timedelta(hours=22), "input_cutoff": dates.tz_localize("UTC") + pd.Timedelta(hours=21), "target_close_at": dates.tz_localize("UTC") + pd.Timedelta(days=1, hours=21), "target_return": np.arange(rows) / 1000, "next_close": 100 + np.arange(rows), "next_open": 100., "close_return": np.arange(rows) / 1000, "momentum_5": .01, "momentum_20": .02, "volume_change": .03, "volume_mean_5": 100., "sentiment_mean": np.arange(rows) / 10, "post_count_log": 1., "unique_authors_log": 1., "volatility_5": .01, "sentiment_std": .1, "sentiment_positive_fraction": .4, "sentiment_neutral_fraction": .2, "sentiment_negative_fraction": .4, "sentiment_change": .01, "sentiment_mean_5": .2, "sentiment_missing": 0, "volatility_regime": "low"})

def test_sequence_windows_return_end_offsets():
    windows, offsets = sequence_windows(np.arange(8).reshape(4, 2), 2)
    assert windows.tolist() == [[[0, 1], [2, 3]], [[2, 3], [4, 5]], [[4, 5], [6, 7]]]
    assert offsets.tolist() == [1, 2, 3]

def test_research_exports_aligned_ablation_dates_and_unshifted_actual(tmp_path):
    frame = research_frame()
    run_research(frame, tmp_path, seed=11, n_splits=2)
    forecasts = pd.read_csv(tmp_path / "forecasts.csv")
    assert forecasts.training_cutoff.notna().all()
    assert (pd.to_datetime(forecasts.training_cutoff, utc=True) < pd.to_datetime(forecasts.issued_at, utc=True)).all()
    for _, group in forecasts.groupby(["ticker", "fold", "model"]):
        assert group.groupby("feature_set").feature_date.apply(lambda x: tuple(x)).nunique() == 1
    assert set(forecasts.actual.dropna().round(6)) <= set(frame.target_return.round(6))
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert {"price_attention_mae", "price_sentiment_mae", "sentiment_mae_improvement_vs_price_attention", "bootstrap_95_low", "bootstrap_95_high"} <= set(comparison.columns)
    assert len(comparison) > 0

def test_optional_xgboost_and_torch_models_run_on_small_folds(tmp_path):
    frame = research_frame(24)
    result = run_research(frame, tmp_path, seed=7, n_splits=1, include_models=True)
    assert "xgboost" in result["models"] and "lstm" in result["models"]
    forecasts = pd.read_csv(tmp_path / "forecasts.csv")
    assert {"xgboost", "lstm"} <= set(forecasts.model)
    import torch
    checkpoint = next((tmp_path / "models").glob("*_lstm_price_sentiment_fold1.pt"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["lookback"] == 5 and "state_dict" in saved and saved["feature_columns"]
    import joblib
    xgb_pipeline = joblib.load(next((tmp_path / "models").glob("*_xgboost_price_sentiment.joblib")))
    assert np.isfinite(xgb_pipeline.predict(frame.iloc[[0]][["close_return", "volatility_5", "momentum_5", "momentum_20", "volume_change", "volume_mean_5", "post_count_log", "unique_authors_log", "sentiment_mean", "sentiment_std", "sentiment_positive_fraction", "sentiment_neutral_fraction", "sentiment_negative_fraction", "sentiment_change", "sentiment_mean_5", "sentiment_missing"]].to_numpy())).all()

def test_forward_forecast_uses_latest_closed_row_without_target_prices(tmp_path):
    input_dir = tmp_path / "inputs"
    make_demo(input_dir, sessions=24, seed=3)
    prices = pd.read_csv(input_dir / "prices.csv")
    posts = pd.read_csv(input_dir / "posts.csv")
    latest = pd.to_datetime(prices.date).max()
    as_of = latest.tz_localize("UTC") + pd.Timedelta(hours=21, minutes=16)
    result = forecast_latest(prices, posts, tmp_path / "forward", as_of=as_of)
    assert len(result) == 5
    assert pd.to_datetime(result.feature_date).dt.date.eq(latest.date()).all()
    assert (pd.to_datetime(result.issued_at, utc=True) <= as_of).all()
    assert (as_of < pd.to_datetime(result.target_open_at, utc=True)).all()
    assert not ({"actual", "actual_return", "target_open", "target_close", "next_open", "next_close", "training_rows"} & set(result.columns))
    assert "current_close" in result.columns


def test_paired_comparison_uses_hand_calculated_same_date_losses():
    rows = []
    dates = pd.date_range("2024-01-02", periods=2)
    for date, actual, baseline, attention, sentiment in zip(dates, [0., 2.], [0., 0.], [0., 1.], [1., 1.]):
        rows.extend([
            {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "naive", "feature_set": "naive", "actual": actual, "prediction": baseline, "sentiment_missing": 0},
            {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "linear", "feature_set": "price_attention", "actual": actual, "prediction": attention, "sentiment_missing": 0},
            {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "linear", "feature_set": "price_sentiment", "actual": actual, "prediction": sentiment, "sentiment_missing": 0},
        ])
    got = _comparisons(pd.DataFrame(rows), 11).iloc[0]
    assert got.price_attention_mae == pytest.approx(.5)
    assert got.price_sentiment_mae == pytest.approx(1.)
    assert got.naive_mae == pytest.approx(1.)
    assert got.sentiment_mae_improvement_vs_price_attention == pytest.approx(-.5)


def test_research_metadata_records_revision_and_local_source_hashes(tmp_path):
    run_research(research_frame(20), tmp_path, seed=5, n_splits=1)
    metadata = __import__("json").loads((tmp_path / "metadata.json").read_text())
    assert metadata["code_revision"]
    assert isinstance(metadata["working_tree_dirty"], bool)
    assert len(metadata["source_hashes"]["research.py"]) == 64

def test_comparison_treats_floating_point_noise_as_no_difference():
    date = pd.Timestamp("2024-01-02")
    rows = [
        {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "naive", "feature_set": "naive", "actual": 0., "prediction": 0., "sentiment_missing": 0},
        {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "linear", "feature_set": "price_attention", "actual": 0., "prediction": 1e-19, "sentiment_missing": 0},
        {"ticker": "XYZ", "fold": 1, "feature_date": date, "model": "linear", "feature_set": "price_sentiment", "actual": 0., "prediction": 0., "sentiment_missing": 0},
    ]
    got = _comparisons(pd.DataFrame(rows), 11).iloc[0]
    assert got.sentiment_mae_improvement_vs_price_attention == 0
    assert np.isnan(got.bootstrap_95_low) and np.isnan(got.bootstrap_95_high)
    assert "insufficient" in got.evidence


def test_next_close_persistence_uses_each_test_row_current_close(tmp_path):
    frame = research_frame(24)
    frame["target_kind"] = "next_close"
    frame["current_close"] = 200 + np.arange(len(frame))
    frame["next_close"] = frame["current_close"] + 1
    run_research(frame, tmp_path, seed=19, n_splits=1)
    forecasts = pd.read_csv(tmp_path / "forecasts.csv")
    baseline = forecasts.loc[forecasts.model.eq("naive")]
    expected = frame.set_index("feature_date").loc[pd.to_datetime(baseline.feature_date), "current_close"].to_numpy()
    np.testing.assert_allclose(baseline.prediction.to_numpy(), expected)
    np.testing.assert_allclose(baseline.current_close.to_numpy(), expected)


def test_minibatches_never_cross_train_validation_boundary():
    from sentiment_stock.research import _training_batches
    assert list(_training_batches(40, split=19, batch_size=16)) == [(0, 16), (16, 19)]


def test_source_hashes_skip_absent_legacy_files(tmp_path):
    from sentiment_stock.research import _source_hashes
    package = tmp_path / "sentiment_stock"
    package.mkdir()
    (package / "research.py").write_text("package source", encoding="utf-8")
    hashes = _source_hashes(package, tmp_path)
    assert len(hashes["research.py"]) == 64
    assert "main.py" not in hashes
    assert "lstm_model.py" not in hashes

def test_mixed_target_kinds_are_rejected():
    frame = research_frame(20)
    frame["target_kind"] = "return"
    frame.loc[frame.index[-1], "target_kind"] = "next_close"
    from sentiment_stock.research import _target_column
    with pytest.raises(ValueError, match="mixed target kinds"):
        _target_column(frame)


def test_too_few_pairs_do_not_claim_a_bootstrap_interval():
    from sentiment_stock.research import _bootstrap_delta
    estimate, low, high = _bootstrap_delta(np.array([.25, .5]), seed=11)
    assert estimate == pytest.approx(.375)
    assert np.isnan(low) and np.isnan(high)



def test_neural_inner_split_reserves_distinct_chronological_validation_tail():
    from sentiment_stock.research import _training_validation_split
    assert _training_validation_split(30) == 25
    assert _training_validation_split(2) == 1
    with pytest.raises(ValueError, match="two windows"):
        _training_validation_split(1)


def test_lstm_saved_normalization_uses_only_inner_optimization_rows(tmp_path):
    pytest.importorskip("torch")
    from sentiment_stock.research import _lstm_predict, LOOKBACK
    train = np.arange(20, dtype=float).reshape(-1, 1)
    train[17:] = 1_000_000.0  # These rows land only in the reserved inner-validation tail.
    checkpoint = tmp_path / "lstm.pt"
    predictions = _lstm_predict(
        train, np.linspace(-.1, .1, len(train)), np.arange(5, dtype=float).reshape(-1, 1),
        seed=13, artifact_path=checkpoint, feature_columns=["x"],
    )
    saved = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
    # 16 optimization windows end at rows 4..16; rows 17..19 are validation-only.
    assert saved["input_mean"].item() == pytest.approx(np.arange(17).mean())
    assert saved["input_medians"].item() == pytest.approx(np.median(np.arange(17)))
    assert len(predictions) == 1 and np.isfinite(predictions).all()
