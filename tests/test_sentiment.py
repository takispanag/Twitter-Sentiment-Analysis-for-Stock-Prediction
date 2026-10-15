import builtins
import os
import pandas as pd
import pytest
import sys
import types
from importlib.metadata import version
from pathlib import Path

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


@pytest.mark.parametrize(
    ("cuda_available", "expected_device", "expected_device_name"),
    [(False, -1, "cpu"), (True, 0, "cuda:0")],
)
def test_twitter_model_uses_a_pinned_revision_and_available_device(
    monkeypatch, cuda_available, expected_device, expected_device_name
):
    captured = {}
    inference_kwargs = {}
    expected_cache = str(Path(__file__).resolve().parents[1] / "artifacts" / "model-cache")
    monkeypatch.delenv("HF_HOME", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "transformers":
            assert os.environ.get("HF_HOME") == expected_cache
        return original_import(name, *args, **kwargs)

    class FakeClassifier:
        model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"))

        def __call__(self, texts, **kwargs):
            inference_kwargs.update(kwargs)
            return [[{"label": "LABEL_0", "score": 0.1}, {"label": "LABEL_1", "score": 0.2}, {"label": "LABEL_2", "score": 0.7}] for _ in texts]

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return FakeClassifier()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="test", pipeline=fake_pipeline))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda_available)))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    scored, metadata = score_posts(pd.DataFrame({"text": ["$AAPL rises"]}), encoder="twitter")

    assert captured["revision"] == "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
    assert captured["device"] == expected_device
    assert metadata["device"] == expected_device_name
    assert metadata["revision"] == captured["revision"]
    assert "January 2018 to December 2021" in metadata["training_corpus_period"]
    assert scored.loc[0, "sentiment_score"] == pytest.approx(0.6)
    assert inference_kwargs["truncation"] is True
    assert inference_kwargs["max_length"] == 128
