# Sentiment Lab

Sentiment Lab is a local-first research project for testing whether finance-related text adds useful information to next-session stock returns. It keeps sentiment classification, return forecasting, historical backtests, and forward paper records as separate questions. It does not place broker orders.

The original project collected posts and compared stock forecasts with several regressors. Its plots did not preserve a reliable time split and label alignment. The current workflow uses timezone-aware inputs, NYSE market sessions, chronological walk-forward folds, training-only transformations, paired date-aligned feature ablations, and a consistent saved forecast schema. The old source and 65 chart images are preserved under `legacy/original/` as historical context.

## Install and run a local demo

Python 3.11 or later is required.

```bash
python -m pip install -e '.[test]'
python -m sentiment_stock demo --output artifacts/demo
streamlit run dashboard.py
python -m pytest
```

The demo writes synthetic prices and posts and runs walk-forward research. The dashboard reads local CSV and JSON artifacts and does not fetch market data, text, or models on page load. You can also use `python -m sentiment_stock generate-demo --output data/demo` to create just the canonical input files. Synthetic results exercise the workflow; they are not evidence of predictive skill or profitability.

The dashboard includes Research, Sentiment, Text benchmark, Backtest, Paper, Legacy comparison, and About views. Select a saved run, ticker, model, and date range in the sidebar. The Backtest view uses the shared simulator and exports its equity path and trades. The Text benchmark view reads `metrics.csv`, `per_class.csv`, `confusion_matrix.csv`, `predictions.csv`, and `metadata.json` from `artifacts/sentiment-benchmark/`.

## Install optional model support

XGBoost and PyTorch are not needed for the standard offline workflow. Add them only for their optional research models:

```bash
python -m pip install -e '.[test,models]'
python -m sentiment_stock demo --output artifacts/demo-models --include-models
```

Transformer encoders are separate and load only when explicitly selected. Install the NLP extra before using them:

```bash
python -m pip install -e '.[nlp]'
python -m sentiment_stock run --prices data/prices.csv --posts data/posts.csv --encoder finbert --output artifacts/finbert-run
```

For an NVIDIA GPU, install the CUDA-enabled PyTorch wheel for your Windows/Python/CUDA setup before installing the NLP extra; the [official PyTorch selector](https://pytorch.org/get-started/locally/) provides the matching command. Verify access with `python -c "import torch; print(torch.cuda.is_available())"`. Transformer scoring automatically uses `cuda:0` when CUDA is available and records the selected device in run metadata; otherwise it runs on CPU.

FinBERT and Twitter RoBERTa are pretrained on different corpora and periods; domain, class mapping, license, and temporal mismatch limit what their output scores mean. Record the model revision and dataset provenance with each benchmark. Model downloads are explicit commands and should be reviewed under each model and dataset's terms. The default `provided` encoder preserves local scores; `vader` is an offline lexicon baseline.

## Inputs and data provenance

Use canonical prices with ticker, session date, open, high, low, close, and volume. Posts need a ticker, text, identifier where available, timestamp, and optional score or author. The adapters also recognize the original column names. All intraday timestamps must carry timezone information. Date-only post records conservatively become available at midnight UTC after that whole calendar day. Price bars are aligned to NYSE session boundaries, including early closes.

### Original Tweet.csv and Company_Tweet.csv source

The original project used Kaggle's large `Tweet.csv` export plus `Company_Tweet.csv` to map tweet IDs to tickers. Download those files separately and provide both local paths:

```bash
python -m sentiment_stock import-original --tweets /path/to/Tweet.csv --mapping /path/to/Company_Tweet.csv --output data/original-import
```

The join preserves IDs and available timestamps. Keep the raw download and generated corpus outside version control. Importing the source does not make its sentiment values expert labels; inspect its date fields, duplicate mapping, and label construction before drawing conclusions.

### Newer 2021–2022 Equinxx corpus

The newer Kaggle [**Stock Tweets for Sentiment Analysis and Prediction**](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction) corpus includes timestamped company tweets and daily OHLCV for 2021–2022. Import a local download with:

```bash
python -m sentiment_stock import-equinxx-tweets --tweets /path/to/stock_tweets.csv --prices /path/to/stock_yfinance_data.csv --output data/equinxx
```

The posts have no gold sentiment labels; VADER, FinBERT, or another scorer provides model outputs rather than truth. Do not use a sentiment scorer's output as a benchmark label, and check Kaggle terms for the exact download before use.

The bundled historical recovery path reads known source CSVs from the preserved git history into an ignored local folder:

```bash
python -m sentiment_stock recover-history --repo . --output data/recovered
python -m sentiment_stock run --prices data/recovered/prices.csv --posts data/recovered/posts.csv --encoder provided --output artifacts/historical
```

That archive is a processed, non-neutral subset with date-only tweet timestamps and incomplete date coverage. Its conservative availability assumption does not prove point-in-time availability. It must not be interpreted as complete Twitter coverage or a clean real-time backtest.

### Market prices, X, and local inputs

Price fetching is an explicit network operation and writes adjusted Yahoo Finance bars:

```bash
python -m sentiment_stock fetch-prices --tickers AAPL MSFT --start 2018-01-01 --end 2024-01-01 --output data/prices
```

X collection is optional and requires an API bearer token in `X_BEARER_TOKEN`. It is bounded by the requested period and page cap; the token is not printed:

```bash
export X_BEARER_TOKEN='your-local-token'
python -m sentiment_stock collect-x --query 'Tesla OR TSLA' --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z --ticker TSLA --output data/x-tsla --max-pages 5
```

Collection and price fetching never happen automatically from the dashboard. Do not commit tokens, private datasets, model caches, or local generated artifacts.

## Research and saved artifacts

Run walk-forward research on local canonical files:

```bash
python -m sentiment_stock run --prices data/prices/prices.csv --posts data/posts.csv --encoder provided --target return --splits 3 --seed 11 --output artifacts/research
```

Each run includes `features.csv`, `forecasts.csv`, `metrics.csv`, `regime_metrics.csv`, `comparison.csv`, `posts_sample.csv`, and `metadata.json`. Forecast rows share the CLI schema used by the dashboard and simulator. The primary target is next-session open-to-close return, `next_close / next_open - 1`. A separate `--target next_close` experiment models the old close-price target and is ineligible for the trading simulator. A volatility experiment is also separate.

Compare naive, price, post-attention, price-attention, and price-sentiment feature sets on matched held-out dates. The sentiment comparison is paired against the same model with price and post attention, with time-block bootstrap uncertainty. Negative MAE improvement means the sentiment variant performed worse. Results depend on data coverage, label quality, model choice, and a finite retrospective period; they do not establish that sentiment adds durable signal.

## Published-label text benchmark

Classification metrics in the text benchmark do not measure return forecasting. You can run a public dataset download explicitly:

```bash
python -m sentiment_stock benchmark-sentiment --source tfns --output artifacts/sentiment-benchmark
```

Or point to a local CSV with text and label columns:

```bash
python -m sentiment_stock benchmark-sentiment --input local-labels.csv --text-column text --label-column label --output artifacts/sentiment-benchmark
```

The benchmark compares majority and fixed VADER baselines with train-only TF-IDF/logistic regression. Optional frozen FinBERT and Twitter RoBERTa inference is opt-in, CPU batched, and cached. First use downloads model files; review the model revisions, source licenses, and corpus period. Saved predictions are row-aligned across each model's common evaluation rows and include per-class scores and confusion counts.

For the mixed-source 38k corpus, use the source-held-out and fine-tuning commands below, then merge runs only if they contain exactly the same test example IDs. The recorded experiment found frozen FinBERT (macro F1 0.663) stronger than one-epoch full-source fine-tuning (0.548) on the TFNS source holdout. See [the measured benchmark report](docs/results/finance-sentiment-benchmark.md); these scores are label-transfer metrics and do not establish stock-return predictiveness.

The accessible finance-text datasets contain published or user-supplied labels, not necessarily qualified expert annotations. Treat scores as agreement with those labels. To prepare a blinded two-annotator finance-direction pilot, use:

```bash
python -m sentiment_stock prepare-sentiment-annotations --posts data/posts.csv --output artifacts/annotation-pilot --sample-size 500 --seed 42
```

An annotation file is not an expert gold set until annotator qualifications, independent agreement, and adjudication have been documented. The separate FinTweet 2025 BUY/HOLD/SELL track targets price movement labels; it is not a human sentiment benchmark and its predictive claims require point-in-time text and split leakage review.

For a stronger source-transfer benchmark, use the 38k-row finance-text compilation after downloading its local Parquet from [Hugging Face](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment). The importer writes a source/class audit and applies the dataset card's explicit `0=neutral, 1=bullish, 2=bearish` mapping. It deduplicates normalized text and holds the TFNS source group out of supervised training. The data has no ticker or timestamp columns, so it cannot establish stock prediction. The card declares MIT while referring users to original constituent terms; review those terms before redistribution.

```bash
python -m sentiment_stock import-finance-corpus --input data/external/financial-tweets-sentiment.parquet --output data/external/financial-tweets-audited
python -m sentiment_stock benchmark-finance-corpus --input data/external/financial-tweets-audited/labeled_posts.csv --audit data/external/financial-tweets-audited/source_audit.json --models majority,vader,tfidf_logreg --output artifacts/sentiment-benchmark/financial-source-test
python -m sentiment_stock finetune-finance-sentiment --input data/external/financial-tweets-audited/labeled_posts.csv --audit data/external/financial-tweets-audited/source_audit.json --output artifacts/sentiment-benchmark/finbert-finetuned
python -m sentiment_stock merge-benchmarks --inputs artifacts/sentiment-benchmark/financial-source-test artifacts/sentiment-benchmark/financial-transformers artifacts/sentiment-benchmark/finbert-finetuned --output artifacts/sentiment-benchmark/financial-comparison
```

FinBERT fine-tuning uses source-group validation, class-weighted loss, early stopping, and one final source-held-out test; run it with the `nlp` extra and a suitable GPU if available. ProsusAI/FinBERT weights are CC BY-NC-SA 4.0 with non-commercial restrictions. The FinTwitBERT sentiment checkpoint is excluded from this holdout benchmark because its model card reports training on this same 38k compilation. The dashboard charts source coverage, class balance, per-class F1, confusion matrices, Brier score, and expected calibration error.

## Forward forecasts and paper ledger

Create an eligible next-session forecast from local data. The `--as-of` timestamp must be timezone-aware and before the next market open:

```bash
python -m sentiment_stock forecast --prices data/prices/prices.csv --posts data/posts.csv --output artifacts/forward --as-of 2024-06-03T21:16:00Z
```

Record only forecast rows that are eligible at the current clock, then settle after target close with canonical prices:

```bash
python -m sentiment_stock paper-record --forecasts artifacts/forward/forecast.csv --db artifacts/paper.sqlite --threshold 0 --cost-bps 5 --slippage-bps 5
python -m sentiment_stock paper-status --db artifacts/paper.sqlite
python -m sentiment_stock paper-settle --prices data/prices/prices.csv --db artifacts/paper.sqlite
```

The SQLite ledger is append-only: exact retries are idempotent, changed identities fail, realized target values are forbidden at record time, and strategy costs are fixed in the record. Historical forecasts cannot be inserted into this forward ledger. Paper records are not broker orders.

## Backtest assumptions and limits

The transparent strategy enters long at the next session's open when predicted return clears a preselected nonnegative threshold, then exits at that session's close. Otherwise it stays in cash. Missing ticker-session forecasts stay in cash. The configured fee and slippage are charged on entry and exit. This assumes fills at the session open and close, simplified liquidity, and no market impact.

The equal-weight buy-and-hold comparator invests at the first eligible open, includes overnight returns through later closes, and is marked at the last available close without a final liquidation charge. This differs from the intraday long/cash strategy. Thresholds must be chosen without tuning on test outcomes. The simulator rejects `next_close` forecasts because they do not specify the required open-to-close target.

## Tests and CI

Run the local suite with `python -m pytest`. It covers data timing, causal features, walk-forward split alignment, legacy regressions, trading accounting, paper-record eligibility, and Streamlit AppTest interactions. GitHub Actions tests only Python 3.15, including the optional XGBoost/PyTorch smoke check. Package metadata still allows Python 3.11 and later, but those older versions are not tested in CI.
