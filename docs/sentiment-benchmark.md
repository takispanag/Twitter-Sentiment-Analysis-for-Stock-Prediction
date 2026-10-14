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

## 38k finance-text compilation: source transfer and fine-tuning

The Hugging Face compilation [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment) has 38,091 text/label/source rows from nine constituent datasets. Its card declares `0=neutral`, `1=bullish`, and `2=bearish`; the importer applies this mapping explicitly because it differs from the TFNS source downloader's numeric mapping. The card lists an MIT license while saying it follows original-source terms, so verify constituent rights before redistribution. The compilation has no ticker or timestamp, and some sources may contain finance headlines or other text rather than tweets. Treat labels as published dataset labels, not expert gold.

Import a local Parquet or CSV copy and inspect aggregate source/class counts without emitting source text:

```powershell
sentiment-stock import-finance-corpus --input data/external/financial-tweets-sentiment.parquet --output data/external/financial-tweets-audited
```

Compare majority, VADER, word/character TF-IDF, and optionally frozen FinBERT/Twitter-RoBERTa using TFNS as the held-out constituent source. All repeated normalized text is removed across groups; repeated examples from the held-out source are kept there, and the other seven source groups are split by source into train and validation. These frozen models may have unknown overlap from pretraining, which is recorded as a caveat.

```powershell
sentiment-stock benchmark-finance-corpus --input data/external/financial-tweets-audited/labeled_posts.csv --audit data/external/financial-tweets-audited/source_audit.json --models majority,vader,tfidf_logreg --output artifacts/sentiment-benchmark/financial-source-test
sentiment-stock benchmark-finance-corpus --input data/external/financial-tweets-audited/labeled_posts.csv --audit data/external/financial-tweets-audited/source_audit.json --models finbert,twitter_roberta --batch-size 16 --output artifacts/sentiment-benchmark/financial-transformers
sentiment-stock merge-benchmarks --inputs artifacts/sentiment-benchmark/financial-source-test artifacts/sentiment-benchmark/financial-transformers --output artifacts/sentiment-benchmark/financial-comparison
```

The trainable path fine-tunes ProsusAI FinBERT, retaining its documented positive/negative/neutral head order and mapping to bullish/bearish/neutral. It uses source-group training and validation, class-weighted loss, gradient clipping, validation-loss early stopping, and one final evaluation against the untouched source holdout. Use a GPU for a full run when possible; optional row caps create an explicitly sampled pilot, not a full-corpus result:

```powershell
sentiment-stock finetune-finance-sentiment --input data/external/financial-tweets-audited/labeled_posts.csv --audit data/external/financial-tweets-audited/source_audit.json --output artifacts/sentiment-benchmark/finbert-finetuned
```

Install the `nlp` extra first. The ProsusAI FinBERT weights are CC BY-NC-SA 4.0 and its model card restricts commercial use. The separate FinTwitBERT-sentiment model reports training on this exact 38,091-row compilation plus a synthetic corpus; it is excluded from fair holdout benchmarking because its training examples overlap our test source. Metrics include macro/weighted F1, balanced accuracy, per-class support, confusion counts, multiclass Brier score, and top-label expected calibration error. They describe label transfer only, not return forecasting.
