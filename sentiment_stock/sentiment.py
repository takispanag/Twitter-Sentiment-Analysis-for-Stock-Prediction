"""Offline-first sentiment encoders."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def score_posts(posts: pd.DataFrame, encoder: str = "vader") -> tuple[pd.DataFrame, dict]:
    """Return continuous sentiment scores, keeping one output row per input post."""
    mode = encoder.casefold()
    if mode not in {"provided", "vader", "finbert", "twitter"}:
        raise ValueError("encoder must be provided, vader, finbert, or twitter")
    result = posts.copy()
    model_ids = {"provided": "provided input labels", "vader": "vaderSentiment/vader_lexicon", "finbert": "ProsusAI/finbert", "twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest"}
    metadata = {"mode": mode, "model_id": model_ids[mode], "revision": None, "licensing_caveat": "Review the encoder and model-card license before redistribution; model scores reflect training corpora and may not represent current financial language."}
    if mode == "provided":
        if "sentiment_score" not in result:
            if "sentiment" in result:
                result["sentiment_score"] = result["sentiment"]
            else:
                raise ValueError("provided mode requires sentiment_score")
        scores = pd.to_numeric(result["sentiment_score"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < -1) | (scores > 1)).any():
            raise ValueError("provided sentiment scores must be finite and within [-1, 1]")
        result["sentiment_score"] = scores
        return result, metadata

    texts = result.get("text", pd.Series([""] * len(result), index=result.index)).fillna("").astype(str).tolist()
    if mode == "vader":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)["compound"] for text in texts]
        metadata["package"] = "vaderSentiment"
    else:
        try:
            import transformers
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(f"{mode} scoring requires the optional 'nlp' dependencies (transformers and torch)") from exc
        model_id = model_ids[mode]
        cache = Path(__file__).resolve().parents[1] / "artifacts" / "model-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["TRANSFORMERS_CACHE"] = str(cache)
        classifier = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id, device=-1)
        outputs = classifier(texts, batch_size=32, truncation=True, top_k=None) if texts else []
        scores, probs = [], []
        for output in outputs:
            labels = {str(item["label"]).casefold(): float(item["score"]) for item in output}
            if mode == "finbert":
                positive = labels.get("positive", 0.0)
                negative = labels.get("negative", 0.0)
                neutral = labels.get("neutral", 0.0)
            else:
                positive = labels.get("label_2", labels.get("positive", 0.0))
                neutral = labels.get("label_1", labels.get("neutral", 0.0))
                negative = labels.get("label_0", labels.get("negative", 0.0))
            probs.append((positive, negative, neutral))
            scores.append(positive - negative)
        result[["prob_positive", "prob_negative", "prob_neutral"]] = pd.DataFrame(probs, index=result.index, columns=["prob_positive", "prob_negative", "prob_neutral"])
        metadata["transformers_version"] = transformers.__version__
        metadata["revision"] = getattr(classifier.model.config, "_commit_hash", None) or "model default revision at download time"
    result["sentiment_score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(result["sentiment_score"].to_numpy()).all():
        raise ValueError("encoder returned nonfinite scores")
    return result, metadata
