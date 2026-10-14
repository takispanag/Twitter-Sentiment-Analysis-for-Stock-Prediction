import pandas as pd
import pytest

from sentiment_stock.benchmark import (
    cache_key,
    compute_metrics,
    expert_label_status,
    normalize_labels,
    prepare_annotations,
    prepare_dataset,
    run_benchmark,
    validate_dataset,
    audit_stock_movement_dataset,
    run_stock_movement_text_track,
    import_equinxx_tweets,
    summarize_annotations,
    import_financial_tweets,
    source_holdout_split,
    run_financial_corpus_benchmark,
    finetune_finbert_source_holdout,
    compute_probability_metrics,
    merge_benchmark_runs,
)


def labelled_frame():
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(18)],
        "text": [
            "Shares are soaring after strong earnings", "The outlook is awful and weak", "The company announced its quarterly results",
            "Bullish momentum continues for the stock", "Investors fear a sharp collapse", "The exchange opens at nine",
            "Revenue beat expectations by a wide margin", "Guidance was cut after disappointing sales", "Trading volume was unchanged",
            "Analysts expect the price to rise", "The stock may plunge after losses", "The firm published a filing today",
            "Strong demand should lift shares", "The company faces severe pressure", "The market closed for the holiday",
            "Profits rose and shares rallied", "A downgrade could hurt the stock", "The company reports results on Friday",
        ],
        "label": ["bullish", "bearish", "neutral"] * 6,
    })


def test_label_mapping_accepts_polarity_aliases_and_rejects_unknown_labels():
    assert normalize_labels(pd.Series(["positive", "negative", "neutral"])).tolist() == [
        "bullish", "bearish", "neutral"
    ]
    with pytest.raises(ValueError, match="unknown label"):
        normalize_labels(pd.Series(["optimistic"]))


def test_dataset_validation_rejects_conflicting_ids_and_split_leakage():
    conflicting = labelled_frame()
    conflicting.loc[1, "post_id"] = conflicting.loc[0, "post_id"]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        validate_dataset(conflicting)

    leaking = labelled_frame()
    leaking["split"] = ["train"] * 9 + ["validation"] * 9
    leaking.loc[9, "post_id"] = leaking.loc[0, "post_id"]
    with pytest.raises(ValueError, match="appears in multiple splits"):
        validate_dataset(leaking)


def test_dataset_validation_rejects_normalized_tweet_text_across_splits_with_row_context():
    frame = labelled_frame().iloc[:6].copy()
    frame["split"] = ["train"] * 3 + ["validation"] * 3
    frame.loc[3, "text"] = "  SHARES   are soaring after strong earnings  "

    with pytest.raises(ValueError, match=r"normalized tweet text.*train.*row 0.*validation.*row 3"):
        validate_dataset(frame)


def test_dataset_validation_preserves_duplicate_text_within_a_split():
    frame = labelled_frame().iloc[:6].copy()
    frame["split"] = ["train"] * 3 + ["validation"] * 3
    frame.loc[1, "text"] = frame.loc[0, "text"].upper()

    result = validate_dataset(frame)

    assert len(result) == len(frame)


def test_generated_stratified_split_is_stable_and_keeps_all_rows():
    source = labelled_frame()
    first = prepare_dataset(source, validation_fraction=1 / 3, seed=42)
    second = prepare_dataset(source, validation_fraction=1 / 3, seed=42)
    assert first["split"].tolist() == second["split"].tolist()
    assert len(first) == len(source)
    assert first.groupby("split")["label"].nunique().min() == 3
    assert set(first.loc[first.split.eq("train"), "example_id"]).isdisjoint(
        first.loc[first.split.eq("validation"), "example_id"]
    )


def test_hand_computed_metrics_use_macro_f1_and_balanced_accuracy():
    result = compute_metrics(
        ["bearish", "bearish", "neutral", "bullish"],
        ["bearish", "neutral", "neutral", "bearish"],
    )
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx((1 / 2 + 2 / 3 + 0) / 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_probability_metrics_compute_multiclass_brier_and_top_label_calibration():
    result = compute_probability_metrics(["bearish", "neutral"], [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], bins=2)
    assert result["multiclass_brier"] == pytest.approx((0.06 + 0.38) / 2)
    assert result["top_label_ece"] == pytest.approx(0.35)
    with pytest.raises(ValueError, match="sum to one"):
        compute_probability_metrics(["bearish"], [[0.5, 0.5, 0.5]])


def test_cache_key_tracks_text_revision_tokenizer_and_preprocessing():
    base = cache_key("shares rise", "model", "abc", "abc", "plain-v1")
    assert base != cache_key("shares fall", "model", "abc", "abc", "plain-v1")
    assert base != cache_key("shares rise", "model", "def", "abc", "plain-v1")
    assert base != cache_key("shares rise", "model", "abc", "xyz", "plain-v1")
    assert base != cache_key("shares rise", "model", "abc", "abc", "social-v2")


def test_published_labels_cannot_be_marked_expert_without_protocol_evidence():
    status = expert_label_status({"annotation_provenance": "published", "annotator_count": 0})
    assert status["label_status"] == "published_labels"
    assert status["expert_protocol_met"] is False
    complete = expert_label_status({
        "annotation_provenance": "human_adjudicated",
        "annotator_count": 2,
        "qualified_finance_annotators": True,
        "qualification_statements": ["finance research", "equity analysis"],
        "anonymized_annotator_ids": ["ann-01", "ann-02"],
        "independent_labels_complete": True,
        "raw_agreement": 1.0,
        "cohen_kappa": 1.0,
        "adjudication_complete": True,
        "adjudicated_labels_complete": True,
        "protocol_version": "finance-direction-v1",
    })
    assert complete["expert_protocol_met"] is True
    incomplete = expert_label_status({"annotation_provenance": "human_adjudicated", "annotator_count": 2,
                                      "qualified_finance_annotators": True, "independent_labels_complete": True,
                                      "adjudication_complete": True, "protocol_version": "v1"})
    assert incomplete["expert_protocol_met"] is False


def test_annotation_summary_computes_pre_adjudication_agreement_and_adjudicated_support():
    summary = summarize_annotations(pd.DataFrame({
        "annotator_1_label": ["bearish", "neutral", "irrelevant/unclear"],
        "annotator_2_label": ["bearish", "bullish", "irrelevant/unclear"],
        "adjudicated_label": ["bearish", "neutral", "irrelevant/unclear"],
    }))
    assert summary["raw_agreement"] == pytest.approx(2 / 3)
    assert summary["disagreement_count"] == 1
    assert summary["adjudicated_three_class_support"] == {"bearish": 1, "neutral": 1, "bullish": 0}
    assert summary["expert_protocol_met"] is False


def test_offline_benchmark_is_row_aligned_and_exports_provenance(tmp_path):
    result = run_benchmark(labelled_frame(), tmp_path, models=["majority", "vader", "tfidf_logreg"], seed=7)
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    metadata = __import__("json").loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert result["prediction_rows"] == len(predictions)
    assert set(predictions["model_id"]) == {"majority", "vader", "tfidf_logreg"}
    assert set(predictions["split"]) == {"validation"}
    assert predictions.groupby("example_id").model_id.nunique().eq(3).all()
    assert {"macro_f1", "balanced_accuracy", "prediction_coverage"} <= set(metrics.columns)
    assert "text" not in predictions.columns
    assert metadata["expert_protocol_met"] is False
    assert metadata["label_status"] == "user_supplied_labels"


def test_annotation_preparation_blinds_existing_scores_and_labels(tmp_path):
    posts = labelled_frame().assign(sentiment_score=0.8, model_prediction="bullish", ticker="XYZ")
    result = prepare_annotations(posts, tmp_path, sample_size=8, seed=9)
    batch = pd.read_csv(tmp_path / "annotations.csv")
    assert len(batch) == 8
    assert result["expert_protocol_met"] is False
    assert {"annotation_id", "text_sha256", "text", "ticker", "annotator_1_label", "annotator_2_label", "adjudicated_label"} <= set(batch.columns)
    assert "sentiment_score" not in batch.columns and "model_prediction" not in batch.columns
    assert batch["annotator_1_label"].isna().all() and batch["annotator_2_label"].isna().all()


def movement_frame():
    rows = []
    for split, start, count in (("train", "2025-01-01", 9), ("validation", "2025-01-10", 3), ("test", "2025-01-13", 6)):
        for index in range(count):
            stamp = pd.Timestamp(start, tz="America/New_York") + pd.Timedelta(hours=index)
            rows.append({"timestamp": stamp.isoformat(), "text": f"AAPL earnings update sample {split} {index}",
                         "ticker": "AAPL", "tweet_url": f"https://x/{split}-{index}",
                         "label_1d_3class": ["BUY", "HOLD", "SELL"][index % 3], "split": split,
                         "rsi_14": 50 + index})
    return pd.DataFrame(rows)


def test_stock_movement_audit_blocks_future_outcome_language_and_nonchronological_splits():
    frame = movement_frame()
    frame.loc[frame.split.eq("test").idxmax(), "text"] = "Those puts are now up 140% since I posted it."
    audit = audit_stock_movement_dataset(frame)
    assert audit["eligible"] is False
    assert audit["issue_counts"]["post_timestamp_outcome_language"] == 1
    assert audit["predictive_results_allowed"] is False

    out_of_order = movement_frame()
    out_of_order.loc[out_of_order.split.eq("test").idxmax(), "timestamp"] = "2024-12-01T10:00:00-05:00"
    assert audit_stock_movement_dataset(out_of_order)["eligible"] is False


def test_failed_stock_movement_audit_writes_no_prediction_or_metric_files(tmp_path):
    frame = movement_frame()
    frame.loc[frame.split.eq("test").idxmax(), "text"] = "Those puts are now up 140% since I posted it."
    (tmp_path / "predictions.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text("stale", encoding="utf-8")
    result = run_stock_movement_text_track(frame, tmp_path, models=["majority"])
    assert result["predictive_results_allowed"] is False
    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "metrics.csv").exists()
    metadata = __import__("json").loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["predictive_results_allowed"] is False


def test_stock_movement_audit_rejects_rows_out_of_order_inside_a_split():
    frame = movement_frame()
    test_rows = frame.index[frame.split.eq("test")].tolist()
    frame.loc[test_rows[1], "timestamp"], frame.loc[test_rows[4], "timestamp"] = (
        frame.loc[test_rows[4], "timestamp"], frame.loc[test_rows[1], "timestamp"]
    )
    audit = audit_stock_movement_dataset(frame)
    assert audit["eligible"] is False
    assert audit["issue_counts"]["nonmonotonic_rows_within_split"] == 1


def test_stock_movement_audit_rejects_normalized_text_cross_split_but_allows_within_split():
    frame = movement_frame()
    frame.loc[frame.split.eq("validation").idxmax(), "text"] = "  AAPL   EARNINGS UPDATE SAMPLE train 0 "

    audit = audit_stock_movement_dataset(frame)

    assert audit["eligible"] is False
    issue = next(issue for issue in audit["issues"] if issue["type"] == "duplicate_text_across_splits")
    assert {entry["split"] for entry in issue["rows"]} == {"train", "validation"}
    assert {entry["row"] for entry in issue["rows"]} == {"0", "9"}

    within_split = movement_frame()
    within_split.loc[1, "text"] = "  AAPL   EARNINGS UPDATE SAMPLE train 0 "
    within_split_audit = audit_stock_movement_dataset(within_split)
    assert within_split_audit["issue_counts"].get("duplicate_text_across_splits", 0) == 0


def test_stock_movement_track_fits_text_only_and_scores_temporal_test(tmp_path):
    frame = movement_frame()
    result = run_stock_movement_text_track(frame, tmp_path, models=["majority", "tfidf_logreg"])
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    metadata = __import__("json").loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert result["prediction_rows"] == 12
    assert set(predictions["split"]) == {"test"}
    assert set(metrics["split"]) == {"test"}
    assert metadata["track"] == "exploratory_stock_movement_text"
    assert metadata["predictors"] == ["text"]
    assert metadata["predictive_results_allowed"] is True
    assert "rsi_14" not in predictions.columns


def test_equinxx_local_import_maps_stock_tweets_schema_without_claiming_labels(tmp_path):
    source = tmp_path / "stock_tweets.csv"
    price_source = tmp_path / "stock_yfinance_data.csv"
    pd.DataFrame({"Date": ["2022-09-29 23:41:16+00:00"], "Tweet": ["Company guidance improved"],
                  "Stock Name": ["TSLA"], "Company Name": ["Tesla, Inc."]}).to_csv(source, index=False)
    pd.DataFrame({"Date": ["2022-09-29"], "Stock Name": ["TSLA"], "Open": [280.0], "High": [290.0],
                  "Low": [275.0], "Close": [285.0], "Adj Close": [285.0], "Volume": [1000]}).to_csv(price_source, index=False)
    result = import_equinxx_tweets(source, tmp_path / "imported", prices_path=price_source)
    posts = pd.read_csv(tmp_path / "imported" / "posts.csv")
    prices = pd.read_csv(tmp_path / "imported" / "prices.csv")
    metadata = __import__("json").loads((tmp_path / "imported" / "dataset.json").read_text(encoding="utf-8"))
    assert result["post_rows"] == 1
    assert posts.loc[0, "ticker"] == "TSLA"
    assert posts.loc[0, "text"] == "Company guidance improved"
    assert posts.loc[0, "source"] == "equinxx_kaggle"
    assert prices.loc[0, "ticker"] == "TSLA" and prices.loc[0, "close"] == 285
    assert metadata["labels_present"] is False
    assert metadata["license_status"] == "user must verify Kaggle terms"
    assert metadata["source_files"] == ["stock_tweets.csv", "stock_yfinance_data.csv"]
    assert "price file was imported" in metadata["limitation"]


def test_equinxx_import_metadata_reflects_missing_optional_price_file(tmp_path):
    source = tmp_path / "stock_tweets.csv"
    pd.DataFrame({"Date": ["2022-09-29 23:41:16+00:00"], "Tweet": ["Company guidance improved"],
                  "Stock Name": ["TSLA"]}).to_csv(source, index=False)

    import_equinxx_tweets(source, tmp_path / "imported")
    metadata = __import__("json").loads((tmp_path / "imported" / "dataset.json").read_text(encoding="utf-8"))

    assert metadata["source_files"] == ["stock_tweets.csv"]
    assert "price file was not provided" in metadata["limitation"]


def test_finance_corpus_import_audits_sources_classes_and_missing_prediction_fields(tmp_path):
    source = tmp_path / "financial.csv"
    pd.DataFrame({
        "tweet": ["A positive outlook", "A neutral note", "A negative outlook", "  "],
        "sentiment": [1, 0, 2, 1],
        "url": ["source-a", "source-a", "source-b", "source-b"],
    }).to_csv(source, index=False)

    result = import_financial_tweets(source, tmp_path / "imported")
    posts = pd.read_csv(tmp_path / "imported" / "labeled_posts.csv")
    audit = __import__("json").loads((tmp_path / "imported" / "source_audit.json").read_text(encoding="utf-8"))

    assert result["usable_rows"] == 3
    assert len(posts) == 3
    assert set(posts.label) == {"bearish", "neutral", "bullish"}
    assert posts.set_index("text").label.to_dict() == {"A positive outlook": "bullish", "A neutral note": "neutral", "A negative outlook": "bearish"}
    assert audit["input_rows"] == 4 and audit["blank_rows"] == 1
    assert audit["source_count"] == 2
    assert audit["has_ticker"] is False and audit["has_timestamp"] is False
    assert audit["label_status"] == "published_labels"
    assert "constituent source terms" in audit["license_caveat"]


def test_source_holdout_prevents_duplicate_text_leakage_and_holds_out_whole_sources():
    rows = []
    for source, texts in {
        "tfns": ["shared article", "test bearish", "test neutral", "test bullish"],
        "kaggle-a": ["SHARED   ARTICLE", "train a", "train b", "train c", "train d", "train e"],
        "github-b": ["valid a", "valid b", "valid c", "valid d"],
    }.items():
        for index, text in enumerate(texts):
            rows.append({"text": text, "label": ["bearish", "neutral", "bullish"][index % 3], "source_url": source})

    train, validation, test, audit = source_holdout_split(pd.DataFrame(rows), holdout_source="tfns")

    assert set(test.source_url) == {"tfns"}
    assert set(validation.source_url) == {"kaggle-a"}
    assert set(train.source_url) == {"github-b"}
    normalized = lambda value: " ".join(value.casefold().split())
    assert {normalized(text) for text in train.text}.isdisjoint({normalized(text) for text in test.text})
    assert audit["heldout_text_overlap_rows_removed_from_training"] == 1
    assert audit["heldout_source"] == "tfns"


def test_source_holdout_fails_for_missing_source_or_one_class_evaluation():
    frame = pd.DataFrame({"text": ["one", "two", "three"], "label": ["bearish", "bullish", "neutral"],
                          "source_url": ["a", "b", "b"]})
    with pytest.raises(ValueError, match="held-out source"):
        source_holdout_split(frame, holdout_source="missing")

    frame.loc[2, "source_url"] = "a"
    with pytest.raises(ValueError, match="at least two classes"):
        source_holdout_split(frame, holdout_source="b")


def test_source_heldout_benchmark_scores_named_test_rows_and_records_source_protocol(tmp_path):
    rows = []
    for source, count in (("test-source", 12), ("validation-source", 12), ("train-source", 18)):
        for index in range(count):
            label = ["bearish", "neutral", "bullish"][index % 3]
            text = {"bearish": "losses pressure shares lower", "neutral": "company files routine report",
                    "bullish": "profits growth shares higher"}[label] + f" {source} {index}"
            rows.append({"text": text, "label": label, "source_url": source})
    frame = pd.DataFrame(rows)

    result = run_financial_corpus_benchmark(frame, tmp_path, holdout_source="test-source", models=["majority", "tfidf_logreg"])
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    metadata = __import__("json").loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert result["example_count"] == 12
    assert set(metrics.split) == {"test"}
    assert set(predictions.split) == {"test"}
    assert metadata["evaluation_protocol"] == "source_group_holdout_v1"
    assert metadata["source_holdout"]["heldout_source"] == "test-source"


def test_finbert_finetune_rejects_unsafe_or_unreasonable_training_parameters(tmp_path):
    with pytest.raises(ValueError, match="epochs must be 1-10"):
        finetune_finbert_source_holdout(pd.DataFrame(), tmp_path, epochs=0)


def test_merge_benchmarks_requires_identical_examples_and_keeps_model_rows_aligned(tmp_path):
    frame = labelled_frame()
    run_benchmark(frame, tmp_path / "run-a", models=["majority"], seed=17, validation_fraction=1 / 3)
    run_benchmark(frame, tmp_path / "run-b", models=["vader"], seed=17, validation_fraction=1 / 3)

    result = merge_benchmark_runs([tmp_path / "run-a", tmp_path / "run-b"], tmp_path / "merged")

    predictions = pd.read_csv(tmp_path / "merged" / "predictions.csv")
    assert result["models"] == 2
    assert set(predictions.model_id) == {"majority", "vader"}
    assert predictions.groupby("example_id").model_id.nunique().eq(2).all()

    run_b_predictions = pd.read_csv(tmp_path / "run-b" / "predictions.csv")
    run_b_predictions.loc[0, "example_id"] = "different-evaluation-row"
    run_b_predictions.to_csv(tmp_path / "run-b" / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="identical held-out example IDs"):
        merge_benchmark_runs([tmp_path / "run-a", tmp_path / "run-b"], tmp_path / "invalid-merge")

    run_b_predictions = pd.read_csv(tmp_path / "run-b" / "predictions.csv")
    original = pd.read_csv(tmp_path / "run-a" / "predictions.csv")
    run_b_predictions["example_id"] = original.example_id.to_numpy()
    run_b_predictions.loc[0, "gold_label"] = "bearish" if run_b_predictions.loc[0, "gold_label"] != "bearish" else "neutral"
    run_b_predictions.to_csv(tmp_path / "run-b" / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="identical held-out example IDs, labels, and splits"):
        merge_benchmark_runs([tmp_path / "run-a", tmp_path / "run-b"], tmp_path / "invalid-label-merge")
