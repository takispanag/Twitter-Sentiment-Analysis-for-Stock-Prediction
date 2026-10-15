"""Local-first dashboard for saved sentiment research and paper artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment_stock.demo import make_demo
from sentiment_stock.features import build_features
from sentiment_stock.research import run_research
from sentiment_stock.sentiment import score_posts
from sentiment_stock.data import normalize_posts, normalize_prices
from sentiment_stock.trading import backtest, cost_sensitivity
from sentiment_stock.paper import read_paper


PAGES = ("Research", "Sentiment", "Text benchmark", "Backtest", "Paper", "Legacy comparison", "About")
RESEARCH_FILES = ("forecasts.csv", "metrics.csv", "metadata.json")
BENCHMARK_FILES = ("metrics.csv", "per_class.csv", "confusion_matrix.csv", "predictions.csv", "metadata.json")


def _artifact_root() -> Path:
    return Path(os.environ.get("SENTIMENT_STOCK_ARTIFACTS", "artifacts")).expanduser()


def _stretch() -> dict[str, object]:
    """Use current Streamlit sizing while retaining compatibility with 1.30."""
    version = tuple(int(part) for part in st.__version__.split(".")[:2])
    return {"width": "stretch"} if version >= (1, 49) else {"use_container_width": True}


def _find_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and any((folder / name).exists() for name in RESEARCH_FILES):
            found.append(folder)
    if any((root / name).exists() for name in RESEARCH_FILES):
        found.insert(0, root)
    return found


def _load_run(folder: Path) -> dict[str, object]:
    missing = [name for name in RESEARCH_FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"required research files are missing: {', '.join(missing)}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    forecasts = pd.read_csv(folder / "forecasts.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    if forecasts.empty or not {"ticker", "model", "feature_set", "feature_date", "prediction"}.issubset(forecasts.columns):
        raise ValueError("forecasts.csv is empty or missing required forecast columns")
    if not {"ticker", "model", "feature_set", "mae"}.issubset(metrics.columns):
        raise ValueError("metrics.csv is missing its expected research columns")
    optional = {}
    for name, key in (("comparison.csv", "comparison"), ("regime_metrics.csv", "regimes"), ("posts_sample.csv", "posts")):
        path = folder / name
        if path.is_file():
            optional[key] = pd.read_csv(path)
    return {"path": folder, "metadata": metadata, "forecasts": forecasts, "metrics": metrics, **optional}


def _format_return(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _sentiment_lift_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize observation-weighted held-out MAE change from adding sentiment."""
    columns = ["ticker", "model", "mae_improvement_pct", "baseline_mae", "sentiment_mae"]
    required = {"ticker", "model", "feature_set", "mae"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame(columns=columns)
    paired = metrics.loc[metrics.feature_set.isin(["price_attention", "price_sentiment"])]
    if paired.empty:
        return pd.DataFrame(columns=columns)
    if "n" in paired:
        paired = paired.copy()
        paired["_n"] = pd.to_numeric(paired["n"], errors="coerce").fillna(0)
        paired["_weighted_error"] = pd.to_numeric(paired["mae"], errors="coerce") * paired["_n"]
        means = paired.groupby(["ticker", "model", "feature_set"], as_index=False, dropna=False).agg(
            _weighted_error=("_weighted_error", "sum"), _n=("_n", "sum")
        )
        means["mae"] = (means["_weighted_error"] / means["_n"]).where(means["_n"].gt(0))
    else:
        means = paired.groupby(["ticker", "model", "feature_set"], as_index=False, dropna=False).mae.mean()
    wide = means.pivot(index=["ticker", "model"], columns="feature_set", values="mae")
    if not {"price_attention", "price_sentiment"}.issubset(wide.columns):
        return pd.DataFrame(columns=columns)
    result = wide.dropna(subset=["price_attention", "price_sentiment"]).reset_index()
    baseline = result["price_attention"]
    result["mae_improvement_pct"] = ((baseline - result["price_sentiment"]) / baseline * 100).where(baseline.ne(0))
    result = result.rename(columns={"price_attention": "baseline_mae", "price_sentiment": "sentiment_mae"})
    return result[columns].dropna(subset=["mae_improvement_pct"]).sort_values(["ticker", "model"]).reset_index(drop=True)


def _banner(metadata: dict) -> None:
    kind = str(metadata.get("dataset_type", "unspecified")).casefold()
    raw = metadata.get("dataset_metadata")
    if isinstance(raw, dict):
        kind = str(raw.get("dataset_type", kind)).casefold()
    if "synthetic" in kind or kind == "demo":
        st.warning("Synthetic research demo. These generated observations are for exercising the workflow and are not evidence of predictive or trading performance.")
    elif "recover" in kind or "historical" in kind:
        st.warning("Retrospective recovered subset. The archived posts are processed, non-neutral, and date-only; assumed availability does not establish point-in-time coverage.")


def _render_research(data: dict, forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.subheader("Walk-forward research")
    st.caption("Forecasts are issued 15 minutes after the feature-session close, using inputs through that close. The primary target is next-session open-to-close return.")
    if "comparison" in data and not data["comparison"].empty:
        st.markdown("#### Paired sentiment ablation")
        comparison = data["comparison"]
        columns = [column for column in ("ticker", "model", "fold", "n", "price_attention_mae", "price_sentiment_mae", "sentiment_mae_improvement_vs_price_attention", "bootstrap_95_low", "bootstrap_95_high", "evidence") if column in comparison]
        st.dataframe(comparison[columns], **_stretch(), hide_index=True)
        st.caption("Positive improvement means sentiment had lower MAE than the same model with attention and price features. Intervals use paired time-block bootstrap samples.")
    metric_view = metrics
    metric_view = metric_view.loc[metric_view.ticker.isin(forecasts.ticker.unique())]
    metric_cols = [column for column in ("ticker", "model", "feature_set", "fold", "n", "mae", "rmse", "directional_accuracy", "no_post_share") if column in metric_view]
    st.markdown("#### Model and feature results")
    st.dataframe(metric_view[metric_cols], **_stretch(), hide_index=True)
    lift = _sentiment_lift_table(metric_view)
    if not lift.empty:
        st.markdown("#### Sentiment impact on held-out MAE")
        st.caption("Percent MAE reduction versus the same model using price and attention features. Positive values indicate lower held-out error after adding sentiment; values summarize available walk-forward folds.")
        heatmap_data = lift.pivot(index="model", columns="ticker", values="mae_improvement_pct")
        chart = px.imshow(
            heatmap_data,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels={"x": "Ticker", "y": "Model", "color": "MAE reduction (%)"},
            title="Sentiment impact on held-out MAE",
        )
        chart.update_layout(template="plotly_white", coloraxis_colorbar_title="MAE reduction (%)", margin=dict(l=16, r=16, t=55, b=16))
        st.plotly_chart(chart, **_stretch())
    if "regimes" in data and not data["regimes"].empty:
        regimes = data["regimes"]
        regimes = regimes.loc[regimes.ticker.isin(forecasts.ticker.unique())]
        st.markdown("#### Results by volatility regime")
        st.dataframe(regimes, **_stretch(), hide_index=True)
    selected_model = st.session_state.get("model", forecasts.model.iloc[0])
    model_forecasts = forecasts.loc[forecasts.model.eq(selected_model)]
    feature_sets = sorted(model_forecasts.feature_set.dropna().astype(str).unique())
    if feature_sets:
        default_feature_set = "price_sentiment" if "price_sentiment" in feature_sets else feature_sets[0]
        chart_feature_set = st.selectbox("Forecast chart feature set", feature_sets,
                                         index=feature_sets.index(default_feature_set), key="forecast_feature_set")
        selected = model_forecasts.loc[model_forecasts.feature_set.astype(str).eq(chart_feature_set)]
        st.caption(f"Forecast chart · feature set: {chart_feature_set}")
        if not selected.empty:
            chart = px.line(selected, x="feature_date", y="prediction", color="ticker", title="Saved forecast traces", labels={"feature_date": "Feature session", "prediction": "Predicted next-session return", "ticker": "Ticker"}, color_discrete_sequence=["#157a78", "#c68731", "#31485e", "#b4654a"])
            chart.update_layout(template="plotly_white", legend_title_text="Ticker", margin=dict(l=16, r=16, t=55, b=16))
            st.plotly_chart(chart, **_stretch())
    st.download_button("Download filtered forecasts", forecasts.to_csv(index=False).encode("utf-8"), "forecasts.csv", "text/csv")
    st.download_button("Download model metrics", metrics.to_csv(index=False).encode("utf-8"), "metrics.csv", "text/csv")
    sample = data.get("posts")
    if isinstance(sample, pd.DataFrame) and not sample.empty:
        st.markdown("#### Source post sample")
        st.caption("Examples are shown as supplied by the local dataset. A missing sentiment field means unavailable scoring, not neutral sentiment.")
        st.dataframe(sample, **_stretch(), hide_index=True)


def _render_sentiment(data: dict, forecasts: pd.DataFrame) -> None:
    st.subheader("Sentiment coverage and forecasts")
    st.caption("Sentiment values are model scores. Missing posts and missing scores remain distinct from neutral posts.")
    scoped = forecasts.loc[forecasts.ticker.isin(forecasts.ticker.unique())]
    cover = data["metadata"].get("sentiment_coverage", {})
    if isinstance(cover, dict):
        left, right = st.columns(2)
        left.metric("Scored session share", _format_return(cover.get("scored_session_share")))
        right.metric("No-post session share", _format_return(cover.get("no_post_share")))
        if cover.get("scored_date_range"):
            st.caption(f"Scored dates: {cover['scored_date_range'][0]} to {cover['scored_date_range'][1]}")
    if "sentiment_missing" in scoped.columns:
        coverage = scoped.groupby("ticker", as_index=False).agg(no_post_share=("sentiment_missing", "mean"), sessions=("sentiment_missing", "size"))
        chart = px.bar(coverage, x="ticker", y="no_post_share", title="No-post share in visible forecast rows", labels={"ticker": "Ticker", "no_post_share": "Share of sessions"}, color_discrete_sequence=["#c68731"])
        chart.update_layout(template="plotly_white", yaxis_tickformat=".0%", margin=dict(l=16, r=16, t=55, b=16))
        st.plotly_chart(chart, **_stretch())
    st.markdown("#### Forecast rows")
    st.dataframe(scoped, **_stretch(), hide_index=True)
    st.download_button("Download visible forecasts", scoped.to_csv(index=False).encode("utf-8"), "forecasts.csv", "text/csv")


def _render_benchmark(path: Path) -> None:
    st.subheader("Text classification benchmark")
    st.caption("Classification is separate from return forecasting. Dataset labels are published or locally supplied labels; they are not expert gold sentiment.")
    if path.is_dir() and not all((path / name).is_file() for name in BENCHMARK_FILES):
        candidates = sorted(folder for folder in path.iterdir() if folder.is_dir() and all((folder / name).is_file() for name in BENCHMARK_FILES))
        if candidates:
            selected_name = st.selectbox("Saved benchmark run", [folder.name for folder in candidates], key="benchmark_run")
            path = path / selected_name
    try:
        missing = [name for name in BENCHMARK_FILES if not (path / name).is_file()]
        if missing:
            raise ValueError(f"benchmark artifacts are not present under {path}; expected {', '.join(BENCHMARK_FILES)}")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        metrics = pd.read_csv(path / "metrics.csv")
        per_class = pd.read_csv(path / "per_class.csv")
        confusion = pd.read_csv(path / "confusion_matrix.csv")
        predictions = pd.read_csv(path / "predictions.csv")
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.info(str(exc))
        st.caption("Run `sentiment-stock benchmark-sentiment --help` to prepare a local or public-label benchmark.")
        return
    st.caption(f"Source: {metadata.get('dataset_id', metadata.get('dataset', 'local dataset'))} · label provenance: {metadata.get('annotation_provenance', 'not verified')}")
    st.caption(f"Evaluation protocol: {metadata.get('evaluation_protocol', 'unspecified')} · seed: {metadata.get('seed', 'n/a')}")
    if metadata.get("evaluation_warning"):
        st.info(metadata["evaluation_warning"])
    dataset_audit = metadata.get("dataset_audit", {})
    if not isinstance(dataset_audit, dict):
        dataset_audit = {}
    if dataset_audit.get("limitation"):
        st.warning(dataset_audit["limitation"])
    if dataset_audit.get("sources"):
        source_rows = [{"source": url, **details} for url, details in dataset_audit["sources"].items()]
        source_frame = pd.DataFrame(source_rows).sort_values("rows", ascending=False)
        st.markdown("#### Corpus provenance and label balance")
        source_chart = px.bar(source_frame, x="rows", y="source", orientation="h", color_discrete_sequence=["#157a78"],
                              title="Rows by constituent source", labels={"rows": "Rows", "source": "Dataset source"})
        source_chart.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"}, margin=dict(l=12, r=16, t=48, b=12))
        st.plotly_chart(source_chart, **_stretch())
        distribution = pd.DataFrame([{"label": label, "rows": count} for label, count in dataset_audit.get("class_support", {}).items()])
        if not distribution.empty:
            class_chart = px.bar(distribution, x="label", y="rows", color="label", title="Published label distribution",
                                 labels={"label": "Canonical sentiment label", "rows": "Rows"},
                                 color_discrete_map={"bearish": "#c68731", "neutral": "#708090", "bullish": "#157a78"})
            class_chart.update_layout(template="plotly_white", showlegend=False, margin=dict(l=12, r=16, t=48, b=12))
            st.plotly_chart(class_chart, **_stretch())
    if metadata.get("expert_label_status"):
        st.info(f"Expert-label status: {metadata['expert_label_status']}")
    st.dataframe(metrics, **_stretch(), hide_index=True)
    model_column = "model_id" if "model_id" in metrics else "model"
    if {model_column, "macro_f1"}.issubset(metrics.columns):
        chart = px.bar(metrics, x=model_column, y="macro_f1", color="split" if "split" in metrics else None,
                       title="Macro F1 on the saved evaluation split", labels={model_column: "Model", "macro_f1": "Macro F1"},
                       color_discrete_sequence=["#157a78", "#c68731"])
        chart.update_layout(template="plotly_white", yaxis_range=[0, 1], margin=dict(l=16, r=16, t=55, b=16))
        st.plotly_chart(chart, **_stretch())
    if {model_column, "top_label_ece", "multiclass_brier"}.issubset(metrics.columns):
        st.markdown("#### Probability quality")
        st.caption("Lower Brier score and expected calibration error indicate better calibrated probabilities. Hard-label baselines are represented as one-hot probabilities and are naturally overconfident.")
        st.dataframe(metrics[[model_column, "split", "multiclass_brier", "top_label_ece"]], **_stretch(), hide_index=True)
    st.markdown("#### Per-class scores")
    st.dataframe(per_class, **_stretch(), hide_index=True)
    if {"model_id", "label", "f1"}.issubset(per_class.columns):
        class_chart = px.bar(per_class, x="label", y="f1", color="model_id", barmode="group",
                             title="Per-class F1 on the same held-out examples", labels={"label": "Sentiment", "f1": "F1", "model_id": "Model"},
                             color_discrete_sequence=["#157a78", "#c68731", "#739b95", "#8c627a"])
        class_chart.update_layout(template="plotly_white", yaxis_range=[0, 1], margin=dict(l=16, r=16, t=55, b=16))
        st.plotly_chart(class_chart, **_stretch())
    st.markdown("#### Confusion counts")
    st.dataframe(confusion, **_stretch(), hide_index=True)
    if {"model_id", "gold_label", "predicted_label", "count"}.issubset(confusion.columns):
        models = sorted(confusion.model_id.dropna().astype(str).unique())
        if models:
            selected = st.selectbox("Confusion matrix model", models, key="benchmark_confusion_model")
            matrix = confusion.loc[confusion.model_id.astype(str).eq(selected)].pivot(index="gold_label", columns="predicted_label", values="count").reindex(index=["bearish", "neutral", "bullish"], columns=["bearish", "neutral", "bullish"], fill_value=0)
            heatmap = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Teal",
                                title=f"Confusion matrix · {selected}", labels={"x": "Predicted", "y": "Published label", "color": "Rows"})
            heatmap.update_layout(template="plotly_white", margin=dict(l=16, r=16, t=55, b=16))
            st.plotly_chart(heatmap, **_stretch())
    st.markdown("#### Row-aligned predictions")
    st.dataframe(predictions, **_stretch(), hide_index=True)
    st.download_button("Download benchmark predictions", predictions.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
    st.caption("Frozen transformer predictions require explicit model downloads and carry training-corpus and licensing limits. See README for model revisions and reproduction steps.")


def _render_backtest(forecasts: pd.DataFrame, selected_model: str, selected_tickers: list[str]) -> None:
    st.subheader("Long or cash, next-session backtest")
    st.caption("Signals enter at the next session open and close at that session's close. Fees and slippage apply per side; the buy-and-hold path includes overnight moves.")
    model = st.selectbox("Backtest model", sorted(forecasts.model.dropna().astype(str).unique()), index=sorted(forecasts.model.dropna().astype(str).unique()).index(selected_model) if selected_model in set(forecasts.model.astype(str)) else 0, key="backtest_model")
    feature_options = sorted(forecasts.loc[forecasts.model.eq(model), "feature_set"].dropna().astype(str).unique())
    if not feature_options:
        st.info("The selected model has no feature sets in this run.")
        return
    feature_set = st.selectbox("Feature set", feature_options, index=feature_options.index("price_sentiment") if "price_sentiment" in feature_options else 0, key="backtest_feature_set")
    threshold = st.number_input("Minimum predicted return", min_value=0.0, max_value=1.0, value=0.0, step=0.005, format="%.3f", key="threshold")
    cost_bps = st.number_input("Fee (bps per side)", min_value=0.0, max_value=1000.0, value=5.0, step=1.0, key="cost_bps")
    slippage_bps = st.number_input("Slippage (bps per side)", min_value=0.0, max_value=1000.0, value=5.0, step=1.0, key="slippage_bps")
    chosen = forecasts.loc[forecasts.ticker.isin(selected_tickers) & forecasts.model.eq(model) & forecasts.feature_set.eq(feature_set)]
    try:
        equity, trades, summary = backtest(chosen, model=model, feature_set=feature_set, threshold=threshold, cost_bps=cost_bps, slippage_bps=slippage_bps)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Backtest could not run: {exc}")
        return
    st.caption(f"Threshold: {threshold:.3f} · fee: {cost_bps:g} bps per side · slippage: {slippage_bps:g} bps per side")
    first, second, third, fourth = st.columns(4)
    first.metric("Net return", _format_return(summary["net_return"]))
    second.metric("Gross return", _format_return(summary["gross_return"]))
    third.metric("Buy-and-hold", _format_return(summary["benchmark_return"]))
    fourth.metric("Trades", f"{summary['trades']:,}")
    equity_chart = px.line(equity, x="session", y=[column for column in ("net_equity", "gross_equity", "benchmark_equity") if column in equity], title="Portfolio value", labels={"session": "Target session", "value": "Equity", "variable": "Series"}, color_discrete_sequence=["#157a78", "#739b95", "#c68731"])
    equity_chart.update_layout(template="plotly_white", margin=dict(l=16, r=16, t=55, b=16))
    st.plotly_chart(equity_chart, **_stretch())
    if not trades.empty:
        st.markdown("#### Executed trades")
        st.dataframe(trades, **_stretch(), hide_index=True)
    st.markdown("#### Fee sensitivity")
    sensitivity = cost_sensitivity(chosen, model=model, feature_set=feature_set, threshold=threshold, slippage_bps=slippage_bps)
    st.dataframe(sensitivity, **_stretch(), hide_index=True)
    st.download_button("Download equity path", equity.to_csv(index=False).encode("utf-8"), "equity.csv", "text/csv")
    st.download_button("Download trades", trades.to_csv(index=False).encode("utf-8"), "trades.csv", "text/csv")


def _render_paper(root: Path) -> None:
    st.subheader("Forward paper ledger")
    st.caption("The ledger records eligible forecasts before target open and settles them only after target close. It is separate from retrospective backtests.")
    db_path = Path(os.environ.get("SENTIMENT_STOCK_PAPER_DB", str(root / "paper.sqlite")))
    st.caption(f"Database: {db_path}")
    try:
        rows = read_paper(db_path)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read paper ledger: {exc}")
        return
    if rows.empty:
        st.info("No forward paper records yet. Use `sentiment-stock forecast`, followed by `paper-record`, to create eligible local records.")
        return
    if "status" in rows:
        pending = int(rows.status.eq("pending").sum())
        settled = int(rows.status.eq("settled").sum())
        a, b = st.columns(2)
        a.metric("Pending", pending)
        b.metric("Settled", settled)
    st.dataframe(rows, **_stretch(), hide_index=True)
    st.download_button("Download paper ledger view", rows.to_csv(index=False).encode("utf-8"), "paper-ledger.csv", "text/csv")


def _render_legacy(root: Path, data: dict, forecasts: pd.DataFrame) -> None:
    st.subheader("Legacy charts beside corrected forecasts")
    st.warning("Archived train/validation/test plots are historical diagnostics. Their original split and target setup is flawed, so they are not comparable causal test results.")
    corrected = data.get("comparison")
    if isinstance(corrected, pd.DataFrame) and not corrected.empty:
        usable = corrected.loc[corrected.model.isin(forecasts.model.unique())]
        cols = [column for column in ("ticker", "model", "fold", "n", "price_attention_mae", "price_sentiment_mae", "sentiment_mae_improvement_vs_price_attention", "bootstrap_95_low", "bootstrap_95_high", "evidence") if column in usable]
        st.markdown("#### Corrected, date-aligned historical forecasts")
        st.dataframe(usable[cols], **_stretch(), hide_index=True)
    archive = root / "legacy" / "original"
    if not archive.is_dir():
        archive = Path("legacy") / "original"
    charts = sorted(archive.rglob("*.png")) if archive.is_dir() else []
    if not charts:
        st.info("The archived chart images under legacy/original are not available in this checkout.")
        return
    names = [str(chart.relative_to(archive)) for chart in charts]
    selected = st.selectbox("Archived original chart", names, key="legacy_chart")
    image_path = archive / selected
    st.image(str(image_path), caption=f"Original archived chart · {selected}", use_container_width=True)
    st.caption(f"Preserved archive: {archive} · {len(charts)} chart images found")


def _render_about(metadata: dict) -> None:
    st.subheader("How to read this research")
    st.markdown("""
    Forecasts use information available through a completed NYSE session close and are issued 15 minutes later. The main target is the next session's open-to-close simple return. `next_close` experiments are separate and cannot be used by the intraday simulator.

    Walk-forward comparisons share held-out dates and fit transformations on training rows only. The paired sentiment comparison tests whether text helps beyond post-attention features. Intervals and historical metrics describe these datasets; they do not establish future predictive power or profitability.

    Date-only posts become available at the next UTC midnight as a conservative assumption. Processed archival subsets can omit neutral posts and many uncovered dates. Missing sentiment is kept separate from neutral sentiment.

    The backtest enters at the next open, exits at that session's close, and charges fees plus slippage on both sides. Buy-and-hold includes overnight price moves and is marked at the last close without a final liquidation fee. Costs, fills, liquidity, and market impact are simplified.
    """)
    provenance = {key: metadata.get(key) for key in ("run_id", "dataset_type", "seed", "target_definition", "time_assumptions", "input_hash", "code_revision") if key in metadata}
    if provenance:
        st.markdown("#### Current run provenance")
        st.json(provenance)


def main() -> None:
    st.set_page_config(page_title="Sentiment Lab", page_icon="SL", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    :root { --ink: #1c2933; --muted: #596973; --paper: #faf8f3; --teal: #157a78; --amber: #c68731; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #f0eee7; border-right: 1px solid #dedbd1; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    [data-testid="stMetricValue"] { color: var(--teal); }
    </style>
    """, unsafe_allow_html=True)
    st.title("Sentiment Lab")
    st.caption("Local research on market text, session returns, and forward paper records")
    root = _artifact_root()
    page = st.sidebar.selectbox("View", PAGES, key="page")
    runs = _find_runs(root)
    if not runs:
        if page == "Text benchmark":
            _render_benchmark(root / "sentiment-benchmark")
            return
        if page == "Paper":
            _render_paper(root)
            return
        st.info(f"No saved research runs found under `{root}`. Generate the synthetic demo below; it runs locally with no network. To use your own data, prepare canonical price and post CSVs and run `sentiment-stock run`.")
        if st.button("Generate the synthetic demo", type="primary"):
            with st.spinner("Generating local demo and walk-forward forecasts…"):
                data_dir = root / "demo-data"
                run_dir = root / "demo"
                make_demo(data_dir)
                prices = normalize_prices(pd.read_csv(data_dir / "prices.csv"))
                posts = normalize_posts(pd.read_csv(data_dir / "posts.csv"))
                posts, encoder = score_posts(posts, "provided")
                metadata = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
                metadata["encoder"] = encoder
                run_research(build_features(prices, posts), run_dir, dataset_metadata=metadata)
            st.success(f"Saved synthetic demo to {run_dir}. Rerun the page to inspect it.")
        st.markdown("""
        **Quick start**

        1. Install the project with `python -m pip install -e '.[test]'`.
        2. Run `python -m sentiment_stock demo --output artifacts/demo`.
        3. Start this page with `streamlit run dashboard.py`.
        """)
        if not (root / "sentiment-benchmark" / "metadata.json").exists():
            st.caption(f"Text benchmark files can be saved under `{root / 'sentiment-benchmark'}` with the `benchmark-sentiment` CLI.")
        return

    st.sidebar.title("Research controls")
    run_names = [str(path) for path in runs]
    run_name = st.sidebar.selectbox("Saved research run", run_names, key="run_path")
    try:
        data = _load_run(Path(run_name))
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Saved run could not load: {exc}")
        st.caption(f"Selected path: {run_name}. Check that forecasts.csv, metrics.csv, and metadata.json belong to one completed run.")
        if page == "Text benchmark":
            _render_benchmark(root / "sentiment-benchmark")
        elif page == "Paper":
            _render_paper(root)
        return
    st.caption(f"Run: {data['path']}")
    metadata = data["metadata"]
    dataset_metadata = metadata.get("dataset_metadata", {})
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}
    dataset_type = metadata.get("dataset_type", dataset_metadata.get("dataset_type", "user_supplied"))
    encoder = dataset_metadata.get("encoder", {})
    encoder_name = encoder.get("mode", "unspecified") if isinstance(encoder, dict) else str(encoder)
    st.caption(f"Provenance · dataset: {dataset_type} · seed: {metadata.get('seed', 'unknown')} · run ID: {metadata.get('run_id', 'unknown')} · encoder: {encoder_name}")
    _banner(metadata)
    forecasts = data["forecasts"].copy()
    metrics = data["metrics"].copy()
    ticker_options = sorted(forecasts.ticker.dropna().astype(str).unique())
    tickers = st.sidebar.multiselect("Tickers", ticker_options, default=ticker_options[:1], key="tickers")
    if not tickers:
        tickers = ticker_options
    st.caption(f"Selected tickers: {', '.join(tickers)}")
    forecasts = forecasts.loc[forecasts.ticker.astype(str).isin(tickers)].copy()
    metrics = metrics.loc[metrics.ticker.astype(str).isin(tickers)].copy()
    if forecasts.empty:
        st.info("No saved forecasts match the selected ticker. Choose another ticker or run.")
        return
    models = sorted(forecasts.model.dropna().astype(str).unique())
    default_model = "linear" if "linear" in models else models[0]
    model = st.sidebar.selectbox("Model", models, index=models.index(default_model), key="model")
    if "feature_date" in forecasts:
        dates = pd.to_datetime(forecasts.feature_date, errors="coerce").dropna()
        if len(dates):
            selected_range = st.sidebar.date_input("Feature date range", value=(dates.min().date(), dates.max().date()), key="date_range")
            if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
                start, end = selected_range
                day = pd.to_datetime(forecasts.feature_date, errors="coerce").dt.date
                forecasts = forecasts.loc[day.between(start, end)]
                metrics = metrics.loc[metrics.ticker.isin(forecasts.ticker.unique())]
    if page == "Research":
        _render_research(data, forecasts, metrics)
    elif page == "Sentiment":
        _render_sentiment(data, forecasts)
    elif page == "Text benchmark":
        _render_benchmark(root / "sentiment-benchmark")
    elif page == "Backtest":
        _render_backtest(forecasts, model, tickers)
    elif page == "Paper":
        _render_paper(root)
    elif page == "Legacy comparison":
        _render_legacy(Path(__file__).resolve().parent, data, forecasts)
    else:
        _render_about(metadata)


if __name__ == "__main__":
    main()
