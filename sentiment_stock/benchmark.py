"""Reproducible, label-provenance-aware finance-post classification benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
import re
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, cohen_kappa_score)
from .data import normalize_posts, normalize_prices


CLASSES = ("bearish", "neutral", "bullish")
LABEL_MAP = {
    "bearish": "bearish", "negative": "bearish", "neg": "bearish", "0": "bearish",
    "neutral": "neutral", "mixed": "neutral", "2": "neutral",
    "bullish": "bullish", "positive": "bullish", "pos": "bullish", "1": "bullish",
}
TFNS_DATASET = "zeroshot/twitter-financial-news-sentiment"
TFNS_URL = f"https://huggingface.co/datasets/{TFNS_DATASET}"
TFNS_LICENSE = "MIT (as stated on the dataset card)"
PREPROCESSING_VERSION = "finance-sentiment-benchmark-v1"
MODEL_IDS = {
    "majority": "majority-class-from-training-split",
    "vader": "vaderSentiment/vader_lexicon",
    "tfidf_logreg": "word-char-tfidf+balanced-logistic-regression",
    "finbert": "ProsusAI/finbert",
    "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}
FINTWEET_DATASET = "michael-zus/fintweet-sentiment-2025"
FINTWEET_URL = f"https://huggingface.co/datasets/{FINTWEET_DATASET}"


def normalize_labels(labels: Iterable[object]) -> pd.Series:
    """Map common polarity names and TFNS class IDs to the canonical three classes."""
    normalized = []
    for value in labels:
        key = str(value).strip().casefold()
        if key not in LABEL_MAP:
            raise ValueError(f"unknown label: {value!r}; expected bearish, neutral, or bullish")
        normalized.append(LABEL_MAP[key])
    return pd.Series(normalized, index=getattr(labels, "index", None), dtype="object")


def _normalized_tweet_text(text: object) -> str:
    """Canonicalize superficial Unicode, casing, and whitespace differences."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize labeled posts while preserving their display text.

    Exact normalized text may repeat within a split and those rows are retained;
    the same text may not appear in more than one supplied split.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("labelled input must be a pandas DataFrame")
    result = frame.copy()
    if "text" not in result or "label" not in result:
        raise ValueError("labelled input requires text and label columns")
    text = result["text"]
    if text.isna().any() or text.astype(str).str.strip().eq("").any():
        raise ValueError("text rows must be nonempty")
    result["text"] = text.astype(str)
    result["label"] = normalize_labels(result["label"])
    if "split" in result:
        result["split"] = result["split"].astype(str).str.strip().str.casefold().replace({"val": "validation", "valid": "validation"})
        if not result["split"].isin(["train", "validation"]).all():
            raise ValueError("split values must be train or validation")
        normalized_text = result["text"].map(_normalized_tweet_text)
        text_splits = pd.DataFrame({"normalized_text": normalized_text, "split": result["split"]})
        for normalized, indexes in text_splits.groupby("normalized_text", sort=False).groups.items():
            if text_splits.loc[list(indexes), "split"].nunique() > 1:
                first_index = indexes[0]
                first_split = result.loc[first_index, "split"]
                for index in indexes[1:]:
                    split = result.loc[index, "split"]
                    if split != first_split:
                        raise ValueError(
                            "normalized tweet text appears across splits: "
                            f"{first_split} row {first_index} and {split} row {index}"
                        )

    id_col = "post_id" if "post_id" in result else "example_id" if "example_id" in result else None
    if id_col:
        result[id_col] = result[id_col].astype(str)
        for _, group in result.groupby(id_col, dropna=False, sort=False):
            if len(group) > 1:
                if "split" in group and group["split"].nunique() > 1:
                    raise ValueError("example ID appears in multiple splits")
                if group["text"].nunique() > 1 or group["label"].nunique() > 1:
                    raise ValueError("conflicting duplicate example IDs")
        result = result.drop_duplicates(subset=[id_col], keep="first")
        result["example_id"] = result[id_col]
    else:
        result["example_id"] = [hashlib.sha256(f"{i}\0{text}".encode("utf-8")).hexdigest()[:24] for i, text in enumerate(result["text"])]
    if "split" in result and result.groupby("example_id")["split"].nunique().gt(1).any():
        raise ValueError("example ID appears in multiple splits")
    return result.reset_index(drop=True)


def prepare_dataset(frame: pd.DataFrame, *, validation_fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Normalize input and make a stable stratified split only when no split exists."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    result = validate_dataset(frame)
    if "split" in result:
        if set(result["split"]) != {"train", "validation"}:
            raise ValueError("input split must contain both train and validation rows")
        return result
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=result.index, dtype="object")
    for _, indices in result.groupby("label", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        if len(positions) < 2:
            raise ValueError("each class needs at least two rows for a stratified split")
        count = min(len(positions) - 1, max(1, int(round(len(positions) * validation_fraction))))
        chosen = rng.permutation(positions)[:count]
        split.loc[chosen] = "validation"
    result["split"] = split
    return result


def compute_metrics(gold: Iterable[str], predicted: Iterable[str]) -> dict:
    gold = list(gold)
    predicted = list(predicted)
    if len(gold) != len(predicted) or not gold:
        raise ValueError("metrics require equally sized, nonempty gold and prediction rows")
    if set(gold) - set(CLASSES) or set(predicted) - set(CLASSES):
        raise ValueError("metrics accept only canonical sentiment labels")
    matrix = confusion_matrix(gold, predicted, labels=CLASSES).tolist()
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, predicted, labels=CLASSES, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(gold, predicted)),
        "confusion_matrix": matrix,
    }


def cache_key(text: str, model_id: str, model_revision: str, tokenizer_revision: str, preprocessing_id: str) -> str:
    normalized = text
    payload = "\0".join((normalized, model_id, model_revision, tokenizer_revision, preprocessing_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expert_label_status(metadata: dict | None) -> dict:
    metadata = metadata or {}
    annotator_ids = metadata.get("anonymized_annotator_ids") or []
    qualifications = metadata.get("qualification_statements") or []
    try:
        agreement = float(metadata.get("raw_agreement"))
        kappa = float(metadata.get("cohen_kappa"))
        agreement_valid = np.isfinite([agreement, kappa]).all() and 0 <= agreement <= 1 and -1 <= kappa <= 1
    except (TypeError, ValueError):
        agreement_valid = False
    protocol_met = bool(
        metadata.get("annotator_count", 0) >= 2
        and metadata.get("qualified_finance_annotators") is True
        and len(annotator_ids) >= 2
        and len(qualifications) >= 2
        and metadata.get("independent_labels_complete") is True
        and agreement_valid
        and metadata.get("adjudication_complete") is True
        and metadata.get("adjudicated_labels_complete") is True
        and metadata.get("protocol_version")
    )
    if protocol_met:
        status = "adjudicated_expert_labels"
    elif metadata.get("annotation_provenance") in {"pilot", "human_pilot", "annotation_pending"}:
        status = "annotation_pending"
    elif metadata.get("annotation_provenance") in {"published", "published_labels"}:
        status = "published_labels"
    else:
        status = "user_supplied_labels"
    return {"label_status": status, "expert_protocol_met": protocol_met}


def summarize_annotations(frame: pd.DataFrame, *, anonymized_annotator_ids: list[str] | None = None,
                          qualification_statements: list[str] | None = None,
                          protocol_version: str | None = None) -> dict:
    """Compute independent-label agreement and adjudicated support without exporting identity or text."""
    columns = ("annotator_1_label", "annotator_2_label", "adjudicated_label")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"annotation file missing columns: {', '.join(sorted(missing))}")
    allowed = {*CLASSES, "irrelevant/unclear"}
    annotator_one = frame.annotator_1_label.fillna("").astype(str).str.strip().str.casefold()
    annotator_two = frame.annotator_2_label.fillna("").astype(str).str.strip().str.casefold()
    adjudicated = frame.adjudicated_label.fillna("").astype(str).str.strip().str.casefold()
    for name, values in ((columns[0], annotator_one), (columns[1], annotator_two), (columns[2], adjudicated)):
        invalid = set(values[values.ne("")]) - allowed
        if invalid:
            raise ValueError(f"invalid {name} values: {', '.join(sorted(invalid))}")
    paired = annotator_one.ne("") & annotator_two.ne("")
    if paired.any():
        first, second = annotator_one[paired].tolist(), annotator_two[paired].tolist()
        raw_agreement = float(np.mean(np.asarray(first) == np.asarray(second)))
        kappa_value = float(cohen_kappa_score(first, second, labels=sorted(allowed)))
        kappa = kappa_value if np.isfinite(kappa_value) else None
        disagreements = {}
        for left, right in zip(first, second):
            if left != right:
                key = "|".join(sorted((left, right)))
                disagreements[key] = disagreements.get(key, 0) + 1
    else:
        raw_agreement, kappa, disagreements = None, None, {}
    complete_pairs = bool(len(frame) and paired.all())
    adjudicated_complete = bool(len(frame) and adjudicated.ne("").all())
    support = {label: int(adjudicated.eq(label).sum()) for label in CLASSES}
    ids = [str(value) for value in anonymized_annotator_ids or [] if str(value).strip()]
    qualifications = [str(value).strip() for value in qualification_statements or [] if str(value).strip()]
    metadata = {"annotation_provenance": "human_adjudicated" if adjudicated_complete else "pilot",
                "annotator_count": len(ids), "anonymized_annotator_ids": ids,
                "qualification_statements": qualifications,
                "qualified_finance_annotators": len(qualifications) >= 2,
                "independent_labels_complete": complete_pairs, "raw_agreement": raw_agreement,
                "cohen_kappa": kappa, "adjudication_complete": adjudicated_complete,
                "adjudicated_labels_complete": adjudicated_complete, "protocol_version": protocol_version}
    return {**metadata, **expert_label_status(metadata), "paired_rows": int(paired.sum()),
            "disagreement_count": int(sum(disagreements.values())), "disagreement_counts": disagreements,
            "adjudicated_three_class_support": support}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "sentiment-stock-benchmark/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not download TFNS public dataset data from Hugging Face: {exc}") from exc


def load_tfns() -> tuple[pd.DataFrame, dict]:
    """Explicitly download TFNS rows through the Hugging Face datasets-server API."""
    source_info = _http_json(f"https://huggingface.co/api/datasets/{TFNS_DATASET}")
    source_revision = str(source_info.get("sha", ""))
    if not source_revision:
        raise RuntimeError("TFNS source did not expose an immutable dataset revision")
    query = urllib.parse.urlencode({"dataset": TFNS_DATASET, "revision": source_revision})
    splits = _http_json(f"https://datasets-server.huggingface.co/splits?{query}").get("splits", [])
    names = {row.get("split") for row in splits}
    if not {"train", "validation"} <= names:
        raise RuntimeError(f"TFNS source did not expose train and validation splits; found {sorted(names)}")
    rows = []
    for split in ("train", "validation"):
        offset = 0
        while True:
            params = urllib.parse.urlencode({"dataset": TFNS_DATASET, "config": "default", "split": split,
                                             "revision": source_revision, "offset": offset, "length": 100})
            page = _http_json(f"https://datasets-server.huggingface.co/rows?{params}")
            items = page.get("rows", [])
            for item in items:
                row = item.get("row", item)
                rows.append({"text": row.get("text", row.get("Tweet", "")), "label": row.get("label"),
                             "split": split, "post_id": f"{split}-{item.get('row_idx', offset)}"})
            offset += len(items)
            total = int(page.get("num_rows_total", offset))
            if not items or offset >= total:
                break
    if not rows:
        raise RuntimeError("TFNS download returned no rows")
    frame = validate_dataset(pd.DataFrame(rows))
    raw_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    metadata = {"dataset_id": TFNS_DATASET, "dataset_url": TFNS_URL, "license": TFNS_LICENSE,
                "dataset_revision": source_revision,
                "dataset_sha256": raw_hash, "annotation_provenance": "published", **expert_label_status({"annotation_provenance": "published"})}
    return frame, metadata


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _preprocess_twitter(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "@user", text)
    return text


def _resolve_hf_revision(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi
        return str(HfApi().model_info(model_id).sha)
    except Exception as exc:
        raise RuntimeError(f"could not resolve immutable Hugging Face revision for {model_id}: {exc}") from exc


def _transformer_predict(texts: list[str], model_name: str, *, batch_size: int, cache_dir: Path) -> tuple[list[str], list[list[float]], dict, int, int, float]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("frozen transformer models require the optional 'nlp' dependencies (transformers and torch)") from exc
    model_id = MODEL_IDS[model_name]
    revision = _resolve_hf_revision(model_id)
    preprocessor = _preprocess_twitter if model_name == "twitter_roberta" else lambda value: value
    preprocessing_id = "cardiff-twitter-normalize-v1" if model_name == "twitter_roberta" else "identity-v1"
    tokenizer_revision = revision
    cache_path = cache_dir / f"{model_name}-{revision}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        cache = {}
    keys = [cache_key(preprocessor(text), model_id, revision, tokenizer_revision, preprocessing_id) for text in texts]
    misses = [index for index, key in enumerate(keys) if key not in cache]
    started = time.perf_counter()
    if misses:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(cache_dir / "hf"))
            model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, cache_dir=str(cache_dir / "hf"))
        except Exception as exc:
            raise RuntimeError(f"could not load frozen model {model_id}@{revision}: {exc}") from exc
        model.eval()
        model.to("cpu")
        id2label = {int(key): str(value).casefold() for key, value in model.config.id2label.items()}
        output_names = []
        for index in range(model.config.num_labels):
            value = id2label.get(index, "").casefold()
            if value in {"negative", "bearish"}:
                output_names.append("bearish")
            elif value in {"neutral"}:
                output_names.append("neutral")
            elif value in {"positive", "bullish"}:
                output_names.append("bullish")
            else:
                explicit_order = ["bearish", "neutral", "bullish"]
                if model.config.num_labels != 3:
                    raise RuntimeError(f"{model_id} has unsupported class labels: {id2label}")
                output_names.append(explicit_order[index])
        for offset in range(0, len(misses), batch_size):
            batch_indices = misses[offset:offset + batch_size]
            batch_texts = [preprocessor(texts[index]) for index in batch_indices]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for index, probs in zip(batch_indices, probabilities):
                canonical = np.zeros(3, dtype=float)
                for model_position, label in enumerate(output_names):
                    canonical[CLASSES.index(label)] = float(probs[model_position])
                prediction = CLASSES[int(np.argmax(canonical))]
                cache[keys[index]] = {"predicted_label": prediction, "probabilities": canonical.tolist()}
            # Persist after every batch so interrupted inference resumes without partial JSON.
            _atomic_json(cache_path, cache)
    predictions = [cache[key]["predicted_label"] for key in keys]
    probabilities = [cache[key]["probabilities"] for key in keys]
    elapsed = time.perf_counter() - started
    return predictions, probabilities, {"model_id": model_id, "model_revision": revision,
        "tokenizer_revision": tokenizer_revision, "preprocessing_id": preprocessing_id,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "device": "cpu", "training_corpus_period": "unknown; training overlap cannot be ruled out" if model_name == "finbert" else "approximately 2018-01 through 2021-12 (model card)",
        "cache_file": str(cache_path)}, len(texts) - len(misses), len(misses), elapsed


def _one_hot(label: str) -> list[float]:
    return [1.0 if value == label else 0.0 for value in CLASSES]


def run_benchmark(frame: pd.DataFrame, output: Path, *, models: Iterable[str] = ("majority", "vader", "tfidf_logreg"),
                  seed: int = 42, validation_fraction: float = 0.2, batch_size: int = 16,
                  cache_dir: Path | None = None, dataset_metadata: dict | None = None,
                  input_path: Path | None = None) -> dict:
    """Run each requested classifier on the same validation rows and export artifacts."""
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    selected = list(dict.fromkeys(models))
    unknown = set(selected) - set(MODEL_IDS)
    if unknown:
        raise ValueError(f"unsupported benchmark models: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("at least one model is required")
    started_at = datetime.now(timezone.utc).isoformat()
    data = prepare_dataset(frame, validation_fraction=validation_fraction, seed=seed)
    train = data[data.split.eq("train")].reset_index(drop=True)
    valid = data[data.split.eq("validation")].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError("benchmark needs nonempty train and validation splits")
    if set(train.example_id) & set(valid.example_id):
        raise ValueError("example ID appears in multiple splits")
    y_train = train.label.tolist()
    y_valid = valid.label.tolist()
    text_train = train.text.tolist()
    text_valid = valid.text.tolist()
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir else Path("artifacts/sentiment-benchmark/cache")
    prediction_rows, metric_rows, per_class_rows, confusion_rows = [], [], [], []
    revisions, cache_totals = {}, {"hits": 0, "misses": 0}

    for model_name in selected:
        model_started = time.perf_counter()
        if model_name == "majority":
            counts = pd.Series(y_train).value_counts()
            winner = sorted(counts[counts.eq(counts.max())].index.tolist())[0]
            pred = [winner] * len(valid)
            probabilities = [_one_hot(winner)] * len(valid)
            revision = {"model_id": MODEL_IDS[model_name], "model_revision": "rule-v1", "tokenizer_revision": None, "preprocessing_id": "none"}
            hits = misses = 0
            duration = time.perf_counter() - model_started
        elif model_name == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except ImportError as exc:
                raise RuntimeError("VADER baseline requires vaderSentiment") from exc
            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(text)["compound"] for text in text_valid]
            pred = ["bearish" if score < -0.05 else "bullish" if score > 0.05 else "neutral" for score in scores]
            probabilities = [_one_hot(label) for label in pred]
            revision = {"model_id": MODEL_IDS[model_name], "model_revision": _package_version("vaderSentiment") or "unknown",
                        "tokenizer_revision": None, "preprocessing_id": "identity-v1", "thresholds": {"bearish_below": -0.05, "bullish_above": 0.05}}
            hits = misses = 0
            duration = time.perf_counter() - model_started
        elif model_name == "tfidf_logreg":
            from scipy.sparse import hstack
            word = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1, sublinear_tf=True)
            char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=1, sublinear_tf=True)
            x_train = hstack([word.fit_transform(text_train), char.fit_transform(text_train)]).tocsr()
            x_valid = hstack([word.transform(text_valid), char.transform(text_valid)]).tocsr()
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs")
            classifier.fit(x_train, y_train)
            raw_probabilities = classifier.predict_proba(x_valid)
            pred = classifier.predict(x_valid).tolist()
            probabilities = []
            for row in raw_probabilities:
                mapped = {label: float(prob) for label, prob in zip(classifier.classes_, row)}
                probabilities.append([mapped.get(label, 0.0) for label in CLASSES])
            revision = {"model_id": MODEL_IDS[model_name], "model_revision": "sklearn-fixed-config-v1",
                        "tokenizer_revision": None, "preprocessing_id": "word-1-2-char-3-5-tfidf-v1",
                        "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced", "seed": seed}
            hits = misses = 0
            duration = time.perf_counter() - model_started
        else:
            pred, probabilities, revision, hits, misses, duration = _transformer_predict(text_valid, model_name, batch_size=batch_size, cache_dir=cache_root)
        revisions[model_name] = revision
        cache_totals["hits"] += hits
        cache_totals["misses"] += misses
        metrics = compute_metrics(y_valid, pred)
        metric_rows.append({"model_id": model_name, "split": "validation", **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                            "prediction_coverage": len(pred) / len(valid), "prediction_rows": len(pred), "duration_seconds": duration,
                            "cache_hits": hits, "cache_misses": misses})
        precision, recall, f1, support = precision_recall_fscore_support(y_valid, pred, labels=CLASSES, zero_division=0)
        for label, p, r, f, n in zip(CLASSES, precision, recall, f1, support):
            per_class_rows.append({"model_id": model_name, "split": "validation", "label": label, "precision": p, "recall": r, "f1": f, "support": int(n)})
        for gold_label, row in zip(CLASSES, metrics["confusion_matrix"]):
            for predicted_label, count in zip(CLASSES, row):
                confusion_rows.append({"model_id": model_name, "split": "validation", "gold_label": gold_label, "predicted_label": predicted_label, "count": count})
        for position, (_, example) in enumerate(valid.iterrows()):
            prediction_rows.append({"example_id": example.example_id, "split": "validation", "gold_label": example.label,
                "predicted_label": pred[position], "model_id": model_name,
                "model_revision": revision.get("model_revision"), "tokenizer_revision": revision.get("tokenizer_revision"),
                "text_sha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
                **{f"prob_{label}": probabilities[position][i] for i, label in enumerate(CLASSES)}})

    _atomic_csv(pd.DataFrame(prediction_rows), run_dir / "predictions.csv")
    _atomic_csv(pd.DataFrame(metric_rows), run_dir / "metrics.csv")
    _atomic_csv(pd.DataFrame(per_class_rows), run_dir / "per_class.csv")
    _atomic_csv(pd.DataFrame(confusion_rows), run_dir / "confusion_matrix.csv")
    provenance = dataset_metadata or {}
    status = expert_label_status(provenance)
    source_hash = _sha256_file(Path(input_path)) if input_path else hashlib.sha256(
        data[["example_id", "text", "label", "split"]].to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()
    counts = {split: {"rows": int(len(group)), "class_support": {label: int(group.label.eq(label).sum()) for label in CLASSES}}
              for split, group in data.groupby("split", sort=True)}
    metadata = {
        "dataset_id": provenance.get("dataset_id", "local-labeled-input"), "dataset_url": provenance.get("dataset_url"),
        "license": provenance.get("license"), "dataset_sha256": provenance.get("dataset_sha256", source_hash),
        "local_file_sha256": source_hash if input_path else None, "source_revision": provenance.get("source_revision"),
        **status, "annotation_provenance": provenance.get("annotation_provenance", status["label_status"]),
        "class_mapping": dict(LABEL_MAP), "split_counts": counts, "model_revisions": revisions,
        "preprocessing_version": PREPROCESSING_VERSION, "package_versions": {name: _package_version(name) for name in ["pandas", "scikit-learn", "vaderSentiment", "transformers", "torch"]},
        "device": "cpu", "batch_size": batch_size, "cache": {**cache_totals, "directory": str(cache_root)},
        "seed": seed, "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "validation_warning": "This is a validation split, not an untouched test set; do not tune on it and claim final-test generalization.",
        "interpretation_boundary": "Text sentiment classification metrics do not measure or establish stock-return predictiveness.",
        "training_overlap_caveat": "Unknown pretraining overlap cannot be ruled out for the frozen baselines.",
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    (run_dir / "README.md").write_text(
        "# Finance-related post sentiment benchmark\n\n"
        f"Reproduce with `sentiment-stock benchmark-sentiment --input INPUT.csv --output {run_dir.as_posix()}`.\n\n"
        f"Label status: **{metadata['label_status']}**. {metadata['validation_warning']}\n\n"
        "Frozen transformer baselines are CPU-only, use immutable revisions recorded in metadata, and may have unknown training overlap. "
        "Classification metrics describe label transfer and do not establish return predictiveness. Source text is omitted from prediction exports.\n",
        encoding="utf-8")
    return {"output": str(run_dir), "prediction_rows": len(prediction_rows), "metric_rows": len(metric_rows),
            "example_count": len(valid), "models": selected, "label_status": metadata["label_status"]}


def prepare_annotations(posts: pd.DataFrame, output: Path, *, sample_size: int = 500, seed: int = 42,
                        stratum_column: str | None = None) -> dict:
    """Create a blind, deterministic two-annotator pilot CSV and protocol instructions."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if "text" not in posts:
        raise ValueError("posts file requires a text column")
    source = posts.copy().reset_index(drop=False).rename(columns={"index": "_source_row"})
    if source.text.isna().any() or source.text.astype(str).str.strip().eq("").any():
        source = source[source.text.notna() & source.text.astype(str).str.strip().ne("")].copy()
    if sample_size > len(source):
        raise ValueError(f"sample_size {sample_size} exceeds {len(source)} available text rows")
    rng = np.random.default_rng(seed)
    if stratum_column:
        if stratum_column not in source:
            raise ValueError(f"sampling stratum column not found: {stratum_column}")
        selected = []
        groups = list(source.groupby(stratum_column, dropna=False, sort=True).groups.values())
        allocations = [int(np.floor(sample_size * len(group) / len(source))) for group in groups]
        for index in range(sample_size - sum(allocations)):
            allocations[index % len(allocations)] += 1
        for group, allocation in zip(groups, allocations):
            selected.extend(rng.choice(np.asarray(list(group), dtype=int), size=allocation, replace=False).tolist())
        sample = source.loc[sorted(selected)].copy()
    else:
        sample = source.loc[np.sort(rng.choice(len(source), size=sample_size, replace=False))].copy()
    records = []
    for ordinal, (_, row) in enumerate(sample.iterrows(), start=1):
        text = str(row.text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity_seed = f"{seed}\0{row._source_row}\0{digest}" if "_source_row" in sample.columns else f"{seed}\0{ordinal}\0{digest}"
        annotation_id = hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()[:16]
        record = {"annotation_id": annotation_id, "text_sha256": digest, "text": text}
        for context_col in ("ticker", "created_at", "source"):
            if context_col in sample.columns:
                record[context_col] = row.get(context_col)
        record.update({"annotator_1_label": "", "annotator_2_label": "", "adjudicated_label": ""})
        records.append(record)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(records), destination / "annotations.csv")
    (destination / "annotation-instructions.md").write_text(
        "# Blind finance-direction annotation pilot\n\n"
        "Annotate the stock outlook expressed by the post, not the author's general emotional tone. Choose `bearish`, `neutral`, `bullish`, or `irrelevant/unclear`. "
        "Use neutral when a relevant post expresses no directional outlook. Use irrelevant/unclear when the post does not support a reliable target-specific direction. "
        "Two annotators must label independently before comparing. A third reviewer adjudicates disagreements. Do not infer missing context or consult model scores. "
        "This batch is a pilot until annotator qualifications, independent agreement, and adjudication are documented.\n",
        encoding="utf-8")
    return {"output": str(destination), "annotation_rows": len(records), "seed": seed, "label_status": "annotation_pending", "expert_protocol_met": False}


def import_equinxx_tweets(tweets_path: Path, output: Path, *, prices_path: Path | None = None) -> dict:
    """Map local Equinxx stock CSVs to canonical inputs without fetching or labeling them."""
    source_path = Path(tweets_path)
    raw = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    folded = {str(column).strip().casefold().replace(" ", "_"): column for column in raw.columns}
    aliases = {
        "created_at": ("date", "tweet_at", "created_at", "timestamp"),
        "text": ("tweet", "text", "body"),
        "ticker": ("stock_name", "ticker", "symbol"),
        "author": ("author", "username", "user"),
        "post_id": ("tweet_id", "post_id", "id"),
    }
    mapped = {}
    for target, names in aliases.items():
        original = next((folded[name] for name in names if name in folded), None)
        if original is not None:
            mapped[original] = target
    normalized = raw.rename(columns=mapped)
    missing = {"created_at", "text", "ticker"} - set(normalized.columns)
    if missing:
        raise ValueError(f"Equinxx tweet CSV needs date, tweet, and Stock Name columns; missing {', '.join(sorted(missing))}")
    if "post_id" not in normalized:
        normalized["post_id"] = [hashlib.sha256(f"{date}\0{ticker}\0{text}".encode("utf-8")).hexdigest()[:24]
                                 for date, ticker, text in zip(normalized.created_at, normalized.ticker, normalized.text)]
    normalized["source"] = "equinxx_kaggle"
    posts = normalize_posts(normalized)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(posts, destination / "posts.csv")
    price_hash = None
    if prices_path is not None:
        price_file = Path(prices_path)
        raw_prices = pd.read_csv(price_file)
        price_columns = {str(column).strip().casefold().replace(" ", "_"): column for column in raw_prices.columns}
        price_aliases = {"date": "date", "stock_name": "ticker", "ticker": "ticker", "symbol": "ticker"}
        price_rename = {original: target for alias, target in price_aliases.items() if (original := price_columns.get(alias)) is not None and target not in raw_prices.columns}
        raw_prices = raw_prices.rename(columns=price_rename)
        prices = normalize_prices(raw_prices)
        _atomic_csv(prices, destination / "prices.csv")
        price_hash = _sha256_file(price_file)
    metadata = {
        "dataset_id": "equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
        "source_url": "https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
        "source_files": [source_path.name] + ([price_file.name] if prices_path is not None else []),
        "input_sha256": _sha256_file(source_path), "rows": len(posts),
        "price_input_sha256": price_hash,
        "labels_present": False, "license_status": "user must verify Kaggle terms",
        "limitation": (
            "This import maps timestamped text and ticker; the provided price file was imported, "
            "and the source has no sentiment gold labels."
            if prices_path is not None else
            "This import maps timestamped text and ticker; the optional paired price file was not provided, "
            "and the source has no sentiment gold labels."
        ),
    }
    _atomic_json(destination / "dataset.json", metadata)
    return {"output": str(destination), "post_rows": len(posts), "price_rows": len(prices) if prices_path is not None else 0, "metadata": metadata}


def _movement_labels(values: Iterable[object]) -> pd.Series:
    mapping = {"buy": "bullish", "hold": "neutral", "sell": "bearish", "0": "bullish", "1": "neutral", "2": "bearish"}
    converted = []
    for value in values:
        key = str(value).strip().casefold()
        if key not in mapping:
            raise ValueError(f"unknown stock-movement label: {value!r}")
        converted.append(mapping[key])
    return pd.Series(converted, index=getattr(values, "index", None), dtype="object")


def audit_stock_movement_dataset(frame: pd.DataFrame) -> dict:
    """Conservative point-in-time audit for the separately labeled forward-movement dataset."""
    issues: list[dict] = []
    required = {"timestamp", "text", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"eligible": False, "predictive_results_allowed": False, "issue_counts": {"missing_required_columns": len(missing)},
                "issues": [{"type": "missing_required_columns", "columns": missing}]}
    split = frame["split"].astype(str).str.casefold()
    if not set(split.unique()) >= {"train", "validation", "test"}:
        issues.append({"type": "missing_temporal_splits", "splits": sorted(split.unique().tolist())})
    normalized_text = frame["text"].map(_normalized_tweet_text)
    text_splits = pd.DataFrame({"normalized_text": normalized_text, "split": split}, index=frame.index)
    for _, indexes in text_splits.groupby("normalized_text", sort=False).groups.items():
        rows = text_splits.loc[list(indexes)]
        if rows["split"].nunique() > 1:
            issues.append({"type": "duplicate_text_across_splits",
                           "rows": [{"row": str(index), "split": str(text_splits.loc[index, "split"])}
                                    for index in indexes]})
    timestamps = []
    naive_count = 0
    for index, value in frame["timestamp"].items():
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                naive_count += 1
                stamp = stamp.tz_localize("America/New_York")
            timestamps.append((index, stamp.tz_convert("UTC")))
        except Exception:
            issues.append({"type": "invalid_timestamp", "row": int(index)})
    if naive_count:
        issues.append({"type": "timezone_missing", "count": naive_count})
    valid = pd.Series({index: stamp for index, stamp in timestamps})
    if len(valid):
        boundaries = {}
        for name in ("train", "validation", "test"):
            indexes = frame.index[split.eq(name) & frame.index.isin(valid.index)]
            if len(indexes):
                boundaries[name] = {"min": valid.loc[indexes].min().isoformat(), "max": valid.loc[indexes].max().isoformat()}
                if not valid.loc[indexes].is_monotonic_increasing:
                    issues.append({"type": "nonmonotonic_rows_within_split", "split": name})
        ordered_splits = [name for name in ("train", "validation", "test") if name in boundaries]
        for earlier, later in zip(ordered_splits, ordered_splits[1:]):
            if pd.Timestamp(boundaries[earlier]["max"]) >= pd.Timestamp(boundaries[later]["min"]):
                issues.append({"type": "nonchronological_split_boundary", "earlier": earlier, "later": later})
    future_patterns = [
        r"\b(?:now|currently|already)\s+(?:up|down|increased|dropped|gained|lost)\b.{0,100}\b(?:since|after)\s+(?:i|we)\s+(?:posted|shared|tweeted|bought|sold)",
        r"\b(?:since|after)\s+(?:i|we)\s+(?:posted|shared|tweeted)\b.{0,100}\b(?:up|down|gained|lost|profit|return|\d+\s*%)",
        r"\b(?:those|my|our)\s+(?:calls|puts|shares|position|trade)\s+(?:are|is|were|was)\s+(?:now\s+)?(?:up|down|profitable|in profit)\b",
    ]
    for index, text in frame["text"].items():
        text_value = "" if pd.isna(text) else str(text)
        if any(re.search(pattern, text_value, flags=re.IGNORECASE | re.DOTALL) for pattern in future_patterns):
            issues.append({"type": "post_timestamp_outcome_language", "row": int(index),
                           "text_sha256": hashlib.sha256(text_value.encode("utf-8")).hexdigest()})
    if "tweet_url" in frame:
        mapped = pd.DataFrame({"url": frame["tweet_url"].astype(str), "split": split})
        if mapped.groupby("url").split.nunique().gt(1).any():
            count = int(mapped.groupby("url").split.nunique().gt(1).sum())
            issues.append({"type": "tweet_cross_split_leakage", "count": count})
        for index, row in frame.iterrows():
            match = re.search(r"/status/(\d+)", str(row.get("tweet_url", "")))
            if match and index in valid.index:
                try:
                    snowflake_ms = (int(match.group(1)) >> 22) + 1288834974657
                    snowflake_time = pd.Timestamp(snowflake_ms, unit="ms", tz="UTC")
                    if abs((snowflake_time - valid.loc[index]).total_seconds()) > 120:
                        issues.append({"type": "timestamp_url_mismatch", "row": int(index),
                                       "text_sha256": hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()})
                except (ValueError, OverflowError):
                    issues.append({"type": "invalid_tweet_snowflake", "row": int(index)})
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue["type"]] = counts.get(issue["type"], 0) + int(issue.get("count", 1))
    return {"eligible": not issues, "predictive_results_allowed": not issues, "issue_counts": counts, "issues": issues,
            "timestamp_boundaries_utc": boundaries if len(valid) else {},
            "audit_policy": "Conservative lexical scan for explicit post-publication outcome language; this cannot prove original tweet authenticity or label correctness."}


def load_fintweet_2025() -> tuple[pd.DataFrame, dict]:
    """Explicitly load the exploratory Hugging Face dataset with its supplied temporal splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("loading FinTweet 2025 requires the optional 'datasets' package") from exc
    try:
        cache = Path("artifacts/sentiment-benchmark/cache/datasets").resolve()
        cache.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import HfApi
        source_revision = str(HfApi().dataset_info(FINTWEET_DATASET).sha)
        dataset = load_dataset(FINTWEET_DATASET, revision=source_revision, cache_dir=str(cache))
    except Exception as exc:
        raise RuntimeError(f"could not download {FINTWEET_DATASET}: {exc}") from exc
    frames = []
    split_counts = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            raise RuntimeError(f"FinTweet 2025 source is missing the {split_name} split")
        part = dataset[split_name].to_pandas()
        part["split"] = split_name
        split_counts[split_name] = len(part)
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    if "label_1d_3class" in result:
        result["label"] = _movement_labels(result["label_1d_3class"])
    elif "label" in result:
        result["label"] = _movement_labels(result["label"])
    source_sha = hashlib.sha256(result.to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()
    metadata = {"dataset_id": FINTWEET_DATASET, "dataset_url": FINTWEET_URL, "dataset_revision": source_revision,
                "dataset_sha256": source_sha,
                "license_claim": "MIT (author-uploaded Hugging Face dataset card)", "author": "michael-zus",
                "source_provenance": "Author-uploaded card claims financial Twitter sources and Interactive Brokers market data; independently unverified.",
                "claimed_row_count": 43468, "downloaded_split_counts": split_counts,
                "label_definition": "author-card claim: next-market-open/one-day forward return thresholds of +/-0.5%, mapped BUY/HOLD/SELL",
                "predictor_policy": "text only; timestamp used for audit/split chronology; all other columns excluded"}
    return result, metadata


def run_stock_movement_text_track(frame: pd.DataFrame, output: Path, *, models: Iterable[str] = ("majority", "tfidf_logreg"),
                                  seed: int = 42, dataset_metadata: dict | None = None) -> dict:
    """Fit text-only baselines on train and evaluate once on the source's chronological test split."""
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_stock_movement_dataset(frame)
    _atomic_json(run_dir / "audit.json", audit)
    if not audit["eligible"]:
        for stale in ("predictions.csv", "metrics.csv", "per_class.csv", "confusion_matrix.csv"):
            (run_dir / stale).unlink(missing_ok=True)
        metadata = {"track": "exploratory_stock_movement_text", "dataset_id": FINTWEET_DATASET,
                    "predictive_results_allowed": False, "predictors": ["text"],
                    "blocked_reason": "point-in-time/content audit failed; no predictive results were generated",
                    "audit": audit, **(dataset_metadata or {})}
        _atomic_json(run_dir / "metadata.json", metadata)
        (run_dir / "README.md").write_text("# Exploratory stock-movement text audit\n\nThe audit failed. No predictive metrics or prediction rows were generated. See `audit.json` for issue counts.\n", encoding="utf-8")
        return {"output": str(run_dir), "prediction_rows": 0, "metric_rows": 0, "predictive_results_allowed": False, "audit_issue_count": len(audit["issues"])}
    selected = list(dict.fromkeys(models))
    if not selected or set(selected) - {"majority", "tfidf_logreg", "vader"}:
        raise ValueError("stock-movement text track supports majority, vader, and tfidf_logreg only")
    data = frame.copy().reset_index(drop=True)
    label_col = "label" if "label" in data else "label_1d_3class" if "label_1d_3class" in data else None
    if label_col is None:
        raise ValueError("stock-movement input needs label or label_1d_3class")
    data["canonical_label"] = _movement_labels(data[label_col])
    train = data[data.split.astype(str).str.casefold().eq("train")]
    test = data[data.split.astype(str).str.casefold().eq("test")]
    if train.empty or test.empty:
        raise ValueError("stock-movement track needs nonempty train and test splits")
    train_text, test_text, y_train, y_test = train.text.astype(str).tolist(), test.text.astype(str).tolist(), train.canonical_label.tolist(), test.canonical_label.tolist()
    preds, metrics_rows, per_class_rows, confusion_rows = [], [], [], []
    revisions = {}
    for model_name in selected:
        started = time.perf_counter()
        if model_name == "majority":
            counts = pd.Series(y_train).value_counts()
            winner = sorted(counts[counts.eq(counts.max())].index.tolist())[0]
            labels = [winner] * len(test)
            probabilities = [_one_hot(winner)] * len(test)
            revision = "training-majority-v1"
        elif model_name == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(text)["compound"] for text in test_text]
            labels = ["bearish" if score < -0.05 else "bullish" if score > 0.05 else "neutral" for score in scores]
            probabilities = [_one_hot(label) for label in labels]
            revision = "fixed-vader-thresholds-v1"
        else:
            from scipy.sparse import hstack
            word = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1, sublinear_tf=True)
            char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=1, sublinear_tf=True)
            x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
            x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs")
            classifier.fit(x_train, y_train)
            raw_probs = classifier.predict_proba(x_test)
            labels = classifier.predict(x_test).tolist()
            probabilities = [[{label: float(prob) for label, prob in zip(classifier.classes_, row)}.get(c, 0.0) for c in CLASSES] for row in raw_probs]
            revision = "word-1-2-char-3-5-tfidf+balanced-logreg-v1"
        duration = time.perf_counter() - started
        revisions[model_name] = revision
        metric = compute_metrics(y_test, labels)
        metrics_rows.append({"model_id": model_name, "split": "test", **{key: value for key, value in metric.items() if key != "confusion_matrix"},
                             "prediction_coverage": len(labels) / len(test), "prediction_rows": len(labels), "duration_seconds": duration})
        precision, recall, f1, support = precision_recall_fscore_support(y_test, labels, labels=CLASSES, zero_division=0)
        per_class_rows.extend({"model_id": model_name, "split": "test", "label": label, "precision": p, "recall": r, "f1": score, "support": int(n)}
                              for label, p, r, score, n in zip(CLASSES, precision, recall, f1, support))
        for gold_label, row in zip(CLASSES, metric["confusion_matrix"]):
            confusion_rows.extend({"model_id": model_name, "split": "test", "gold_label": gold_label, "predicted_label": predicted, "count": count}
                                  for predicted, count in zip(CLASSES, row))
        for position, (_, example) in enumerate(test.iterrows()):
            preds.append({"example_id": str(example.get("tweet_url", example.name)), "split": "test", "gold_label": example.canonical_label,
                          "predicted_label": labels[position], "model_id": model_name, "model_revision": revision,
                          "text_sha256": hashlib.sha256(str(example.text).encode("utf-8")).hexdigest(),
                          **{f"prob_{label}": probabilities[position][i] for i, label in enumerate(CLASSES)}})
    _atomic_csv(pd.DataFrame(preds), run_dir / "predictions.csv")
    _atomic_csv(pd.DataFrame(metrics_rows), run_dir / "metrics.csv")
    _atomic_csv(pd.DataFrame(per_class_rows), run_dir / "per_class.csv")
    _atomic_csv(pd.DataFrame(confusion_rows), run_dir / "confusion_matrix.csv")
    metadata = {"track": "exploratory_stock_movement_text", "dataset_id": FINTWEET_DATASET, "dataset_url": FINTWEET_URL,
                "author_claimed_license": "MIT; author attribution required", "source_claims": "Tweet sources and Interactive Brokers data as stated by uploader; not independently verified.",
                "predictive_results_allowed": True, "predictors": ["text"], "excluded_columns": [column for column in frame.columns if column not in {"text", "timestamp", "split", label_col, "tweet_url"}],
                "split_counts": {name: int(data.split.astype(str).str.casefold().eq(name).sum()) for name in ("train", "validation", "test")},
                "test_class_support": {label: int(test.canonical_label.eq(label).sum()) for label in CLASSES}, "model_revisions": revisions,
                "seed": seed, "audit": audit, "label_definition": "BUY/HOLD/SELL are author-claimed forward-return threshold classes; exploratory target, not sentiment gold labels.",
                "interpretation_boundary": "These results are exploratory and do not establish a tradable strategy or return forecast.", **(dataset_metadata or {})}
    _atomic_json(run_dir / "metadata.json", metadata)
    (run_dir / "README.md").write_text("# Exploratory stock-movement text track\n\nOnly tweet text is used by the classifiers. Models train on the supplied chronological training split and are evaluated once on the supplied latest test split; validation is not used. The dataset card's label and source claims have not been independently verified. The output is exploratory and does not establish a tradable strategy.\n", encoding="utf-8")
    return {"output": str(run_dir), "prediction_rows": len(preds), "metric_rows": len(metrics_rows), "predictive_results_allowed": True, "models": selected}
