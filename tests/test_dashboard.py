from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from sentiment_stock.data import normalize_posts, normalize_prices
from sentiment_stock.demo import make_demo
from sentiment_stock.features import build_features
from sentiment_stock.research import run_research
from sentiment_stock.sentiment import score_posts


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard.py"


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    run_path = artifacts / "demo"
    data_path = artifacts / "demo-input"
    make_demo(data_path, sessions=32, seed=11)
    prices = normalize_prices(pd.read_csv(data_path / "prices.csv"))
    posts = normalize_posts(pd.read_csv(data_path / "posts.csv"))
    posts, encoder = score_posts(posts, "provided")
    run_research(
        build_features(prices, posts),
        run_path,
        seed=11,
        n_splits=1,
        dataset_metadata={"dataset_type": "synthetic", "encoder": encoder},
    )
    monkeypatch.setenv("SENTIMENT_STOCK_ARTIFACTS", str(artifacts))
    return artifacts, run_path


def test_dashboard_explains_local_first_setup_when_no_runs_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_STOCK_ARTIFACTS", str(tmp_path / "empty-artifacts"))
    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert any("Generate the synthetic demo" in item.value for item in app.info)
    assert any("no network" in item.value.lower() for item in app.info)


def test_dashboard_loads_run_and_ticker_selection_updates_visible_scope(saved_run):
    _, run_path = saved_run
    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert any("synthetic" in item.value.lower() for item in app.warning)
    assert any(str(run_path) in item.value for item in app.caption)
    assert any("seed: 11" in item.value for item in app.caption)
    ticker_picker = app.multiselect(key="tickers")
    assert "AAPL" in ticker_picker.value
    ticker_picker.unselect("AAPL").select("MSFT").run()
    assert not app.exception
    assert app.multiselect(key="tickers").value == ["MSFT"]
    assert any("MSFT" in item.value for item in app.caption)


def test_dashboard_backtest_controls_drive_accounting_and_paper_does_not_create_db(saved_run, tmp_path):
    artifacts, _ = saved_run
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox(key="page").set_value("Backtest").run()
    assert not app.exception
    baseline = next(item.value for item in app.metric if item.label == "Net return")
    app.number_input(key="threshold").set_value(1.0).run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Net return") != baseline

    missing_db = artifacts / "missing-paper.sqlite"
    app.selectbox(key="page").set_value("Paper").run()
    assert not app.exception
    assert not missing_db.exists()
    assert any("No forward paper records" in item.value for item in app.info)


def test_dashboard_reports_corrupt_saved_run_without_traceback(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    corrupt = artifacts / "broken"
    corrupt.mkdir(parents=True)
    (corrupt / "metadata.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_STOCK_ARTIFACTS", str(artifacts))
    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert any("could not load" in item.value.lower() for item in app.error)


@pytest.mark.parametrize("metadata_json", ["[]", '"metadata is a string"'])
def test_text_benchmark_rejects_non_object_metadata_without_app_exception(tmp_path, monkeypatch, metadata_json):
    artifacts = tmp_path / "artifacts"
    benchmark = artifacts / "sentiment-benchmark"
    benchmark.mkdir(parents=True)
    (benchmark / "metadata.json").write_text(metadata_json, encoding="utf-8")
    for name, contents in {
        "metrics.csv": "model,macro_f1\nmajority,0.3\n",
        "per_class.csv": "model,label,f1\nmajority,bearish,0.3\n",
        "confusion_matrix.csv": "model,true_label,predicted_label,count\nmajority,bearish,bearish,1\n",
        "predictions.csv": "post_id,text,predicted_label\n1,example,bearish\n",
    }.items():
        (benchmark / name).write_text(contents, encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_STOCK_ARTIFACTS", str(artifacts))
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox(key="page").set_value("Text benchmark").run()

    assert not app.exception
    assert any("metadata.json must contain a JSON object" in item.value for item in app.info)


def test_research_forecast_chart_filter_selects_one_feature_set(saved_run):
    app = AppTest.from_file(APP_PATH).run()
    picker = app.selectbox(key="forecast_feature_set")
    assert "price_attention" in picker.options
    assert "price_sentiment" in picker.options

    picker.set_value("price_attention").run()

    assert not app.exception
    assert any("Forecast chart · feature set: price_attention" in item.value for item in app.caption)


def test_dashboard_renders_source_audit_and_calibration_views(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    benchmark = artifacts / "sentiment-benchmark" / "run1"
    benchmark.mkdir(parents=True)
    (benchmark / "metadata.json").write_text(__import__("json").dumps({
        "dataset_id": "test-corpus", "annotation_provenance": "published", "evaluation_protocol": "source_group_holdout_v1",
        "seed": 42, "evaluation_warning": "Held-out source test.",
        "dataset_audit": {"limitation": "No ticker or timestamp columns.", "class_support": {"bearish": 4, "neutral": 3, "bullish": 5},
                          "sources": {"source-a": {"rows": 12}, "source-b": {"rows": 4}}},
    }), encoding="utf-8")
    pd.DataFrame([{"model_id": "tfidf_logreg", "split": "test", "macro_f1": .7, "multiclass_brier": .4, "top_label_ece": .1}]).to_csv(benchmark / "metrics.csv", index=False)
    pd.DataFrame([{"model_id": "tfidf_logreg", "split": "test", "label": label, "precision": .7, "recall": .7, "f1": .7, "support": 2}
                  for label in ("bearish", "neutral", "bullish")]).to_csv(benchmark / "per_class.csv", index=False)
    pd.DataFrame([{"model_id": "tfidf_logreg", "split": "test", "gold_label": gold, "predicted_label": predicted, "count": 1}
                  for gold in ("bearish", "neutral", "bullish") for predicted in ("bearish", "neutral", "bullish")]).to_csv(benchmark / "confusion_matrix.csv", index=False)
    pd.DataFrame([{"example_id": "one", "split": "test", "gold_label": "bearish", "predicted_label": "bearish", "model_id": "tfidf_logreg"}]).to_csv(benchmark / "predictions.csv", index=False)
    monkeypatch.setenv("SENTIMENT_STOCK_ARTIFACTS", str(artifacts))

    app = AppTest.from_file(APP_PATH).run()
    app.selectbox(key="page").set_value("Text benchmark").run()

    assert not app.exception
    assert app.selectbox(key="benchmark_run").value == "run1"
    assert any("No ticker or timestamp" in item.value for item in app.warning)
    assert any("Probability quality" in item.value for item in app.markdown)
