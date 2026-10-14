import pandas as pd
import pytest
from importlib.metadata import version

from sentiment_stock.sentiment import score_posts


def test_vader_assigns_signed_finite_scores_and_keeps_neutral_rows():
    posts = pd.DataFrame({"text": ["I love this company", "This is terrible", "The market is open"]})
    scored, metadata = score_posts(posts)
    assert len(scored) == 3
    assert scored.loc[0, "sentiment_score"] > 0
    assert scored.loc[1, "sentiment_score"] < 0
    assert scored.loc[2, "sentiment_score"] == pytest.approx(0)
    assert scored.sentiment_score.map(pd.notna).all()
    assert metadata["mode"] == "vader"
    assert metadata["package_version"] == version("vaderSentiment")
    assert metadata["revision"] == metadata["package_version"]
    assert metadata["training_corpus_period"].startswith("Unknown")
    assert metadata["temporal_caveat"]


def test_provided_scores_outside_unit_interval_fail():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        score_posts(pd.DataFrame({"sentiment_score": [1.5]}), encoder="provided")


def test_provided_scores_are_preserved():
    scored, metadata = score_posts(pd.DataFrame({"sentiment_score": [-1.0, 0.25]}), encoder="provided")
    assert scored.sentiment_score.tolist() == [-1.0, 0.25]
    assert metadata["mode"] == "provided"
