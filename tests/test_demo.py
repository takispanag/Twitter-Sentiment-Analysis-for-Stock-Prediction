import json

import pandas as pd

from sentiment_stock.demo import make_demo
from sentiment_stock.cli import main


def test_demo_is_repeatable_and_metadata_is_labeled_synthetic(tmp_path):
    first = make_demo(tmp_path / "one", sessions=20, seed=11)
    second = make_demo(tmp_path / "two", sessions=20, seed=11)
    third = make_demo(tmp_path / "three", sessions=20, seed=12)
    one = pd.read_csv(first["prices_path"])
    two = pd.read_csv(second["prices_path"])
    other = pd.read_csv(third["prices_path"])
    pd.testing.assert_frame_equal(one, two)
    assert not one.equals(other)
    metadata = json.loads((tmp_path / "one" / "dataset.json").read_text())
    assert metadata["dataset_type"] == "synthetic"


def test_demo_sentiment_is_independent_of_future_price_labels(tmp_path):
    result = make_demo(tmp_path / "demo", sessions=12, seed=42)
    posts = pd.read_csv(result["posts_path"])
    assert "label" not in posts.columns
    assert "target" not in posts.columns
    assert posts.sentiment_score.abs().max() <= 1


def test_cli_returns_failure_for_invalid_demo_data(tmp_path):
    assert main(["generate-demo", "--sessions", "1", "--output", str(tmp_path / "invalid")]) == 1
