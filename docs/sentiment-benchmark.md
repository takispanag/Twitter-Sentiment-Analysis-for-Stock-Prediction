# Sentiment and stock-movement datasets

The repository has separate workflows for sentiment labels, future stock-movement labels, and the recovered historical subset. Their scores answer different questions and should not be combined.

## TFNS published sentiment labels

The public Twitter Financial News Sentiment (TFNS) dataset has finance-related post text and published `bearish`, `bullish`, and `neutral` labels. Its Hugging Face card states that it was collected through the Twitter API and is MIT-licensed. The card does not document finance-expert annotators or an expert annotation protocol, so benchmark metadata calls these **published labels**. The card’s displayed total and split counts have varied/inconsistencies; use the exact downloaded split counts and dataset hash in each run’s `metadata.json`.

Run the offline baselines without network access after the CSV is available:

```powershell
sentiment-stock benchmark-sentiment --input labeled_posts.csv --text-column text --label-column label --split-column split --output artifacts/sentiment-benchmark/local
```

Download TFNS only when explicitly requested. The default model set is majority, fixed-threshold VADER, and train-only word/character TF-IDF logistic regression. Frozen FinBERT and Twitter RoBERTa are opt-in; they require the optional `nlp` dependencies and network access for the first download. Their immutable model and tokenizer revisions, CPU batching, normalized-text cache keys, and run provenance are written to metadata.

```powershell
sentiment-stock benchmark-sentiment --source tfns --output artifacts/sentiment-benchmark/tfns
sentiment-stock benchmark-sentiment --source tfns --models finbert,twitter_roberta --batch-size 16 --output artifacts/sentiment-benchmark/tfns-transformers
```

The benchmark fits the lexical supervised model on the source `train` split and evaluates every model on the same `validation` rows. Validation is not an untouched final test set. Text is omitted from `predictions.csv`; generated artifacts and model caches stay under ignored `artifacts/`.

## Optional blind annotation pilot

Create a blind two-annotator CSV from a local text-bearing post file. Existing labels and model scores are omitted. `--stratify-column` is optional and should only be used when the chosen field is explicitly a sampling stratum.

```powershell
sentiment-stock prepare-sentiment-annotations --posts posts.csv --sample-size 500 --seed 42 --output artifacts/sentiment-annotations/pilot
```

The generated instructions define stock-direction labels and an `irrelevant/unclear` option. A batch remains `annotation_pending` until two qualified finance annotators independently label the same rows and a third reviewer adjudicates disagreements. Do not call pilot labels expert labels.

## Exploratory FinTweet 2025 stock-movement labels

The separate Hugging Face dataset [`michael-zus/fintweet-sentiment-2025`](https://huggingface.co/datasets/michael-zus/fintweet-sentiment-2025) is described by its uploader as 43,468 rows with train/validation/test temporal splits. Its card claims BUY/HOLD/SELL labels derived from forward returns and an MIT license, and attributes tweet sources and market data to named social accounts and Interactive Brokers. These are uploader claims and have not been independently verified. BUY/HOLD/SELL are **stock-movement labels**, not sentiment gold labels.

The code downloads this source only when `--source fintweet-2025` is explicitly selected. It audits split chronology, timezone-aware timestamps, tweet URL Snowflake timestamps when present, duplicate URL leakage across splits, and explicit post-publication outcome language in text. A failed audit writes `audit.json` and failure metadata and creates no prediction or metric files. The dataset preview includes “those puts are now up 140% since I posted it,” which the audit treats as post-outcome wording and blocks from prediction reporting.

If the audit passes, the exploratory track trains majority/VADER/TF-IDF text-only baselines on the supplied train split, does not use validation for model selection, and evaluates the supplied chronological test split. Timestamp and labels are audit/target fields; ticker, author, category, session, market regime, sector, market-cap buckets, and all numeric indicators are excluded from features.

```powershell
sentiment-stock benchmark-stock-movement --source fintweet-2025 --output artifacts/sentiment-benchmark/fintweet-2025
sentiment-stock benchmark-stock-movement --input local_fintweet.csv --output artifacts/sentiment-benchmark/fintweet-local
```

This track is exploratory. Passing the audit cannot prove that source text is authentic, labels were generated as claimed, or all temporal leakage is absent. The metrics do not establish trading performance.

## Kaggle tweet corpora and recovered history

The older Kaggle corpus the project originally used is [Omer Metin’s “Tweets about the Top Companies from 2015 to 2020”](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020), with `Tweet.csv` and `Company_Tweet.csv`. Import a local copy with the existing ID join:

```powershell
sentiment-stock import-original --tweets Tweet.csv --mapping Company_Tweet.csv --output artifacts/original-import
```

The recovered archive is a processed, date-only non-neutral subset, not the full raw `Tweet.csv` corpus and not an expert sentiment benchmark. Its conservative availability assumption delays each post to the next UTC midnight.

The newer Equinxx Kaggle corpus [“Stock Tweets for Sentiment Analysis and Prediction”](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction) contains `stock_tweets.csv` (`Date`, `Tweet`, `Stock Name`, `Company Name`) and `stock_yfinance_data.csv` (daily OHLCV); published studies describe roughly 80,793 posts from 2021–2022. It provides no gold sentiment labels. Import local copies without auto-download:

```powershell
sentiment-stock import-equinxx-tweets --tweets stock_tweets.csv --prices stock_yfinance_data.csv --output artifacts/equinxx-2021-2022
```

The importer preserves text and timestamps and normalizes tickers/prices. Kaggle terms should be checked for the user’s exact download before use; do not redistribute the source corpus. For this corpus, VADER or a frozen encoder produces model scores, not truth labels.

These newer datasets do not replace the recovered historical comparison. Return forecasting remains a separate task with its own causal features, target definitions, and forecast artifacts.
