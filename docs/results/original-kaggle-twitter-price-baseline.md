# Original Kaggle tweets: first chronological price baseline

This is an exploratory stock-return experiment using the recovered Kaggle tweet archive. It does not establish that tweet sentiment is profitable or causally predictive.

## Data audit

The downloaded archive contained `Tweet.csv`, `Company_Tweet.csv`, and `Company.csv`. The importer joined tweet IDs to ticker mappings and produced 4,336,445 tweet-to-company rows across AAPL, AMZN, GOOG, GOOGL, MSFT, and TSLA. There were no missing text or timestamp values in the imported rows. Tweet timestamps span 2015-01-01 through 2019-12-31. The source has no expert sentiment labels; the importer preserved the original tweet text and timestamps and assigned no gold labels.

Daily adjusted OHLCV bars were fetched separately for the six symbols. The price panel has 7,560 ticker-session rows from 2015-01-02 through 2020-01-03. Source files and generated row-level outputs remain local and ignored by Git. Review Kaggle's terms and Yahoo Finance's data terms before redistributing data.

## Protocol

VADER compound sentiment was scored for every imported tweet. Features for each prediction use only posts available through that session's close; the forecast is issued 15 minutes after close and the target is the next NYSE session's open-to-close return. Posts were assigned their source timestamp as availability time. The experiment used three expanding chronological folds independently per ticker, with 311 test sessions per fold and ticker (5,598 evaluated ticker-sessions per model/feature set). The prediction models were linear regression and linear SVR. Benchmarks include a training-mean naive return, price-only features, price plus tweet volume/author attention, and price plus VADER sentiment. No random split or test-driven model selection was used.

This is a retrospective experiment. VADER is a generic lexicon baseline, not a financial-domain classifier. Tweet availability, completeness, and timestamp authenticity are inherited from the downloaded source; adjusted historical prices may also be revised. The result is a research baseline, not a live-trading claim.

## Results

Mean absolute error (MAE) and root mean squared error (RMSE) are averages across ticker/fold rows; lower is better. Directional accuracy measures the sign of predicted versus realized returns.

| Model | Feature set | MAE | RMSE | Directional accuracy |
|---|---|---:|---:|---:|
| Naive training-mean return | Naive | 0.009689 | 0.013348 | 0.5064 |
| Linear regression | Price only | 0.009763 | 0.013402 | 0.5105 |
| Linear regression | Price + attention | 0.009795 | 0.013431 | 0.5157 |
| Linear regression | Price + VADER sentiment | 0.009927 | 0.013561 | 0.5089 |
| Linear SVR | Price only | 0.009916 | 0.013604 | 0.5080 |
| Linear SVR | Price + attention | 0.010032 | 0.013680 | 0.5032 |
| Linear SVR | Price + VADER sentiment | 0.010268 | 0.013951 | 0.4954 |

Adding VADER sentiment did not improve the average MAE over price plus attention: MAE rose by 0.000131 for linear regression and 0.000236 for linear SVR. Across 18 paired ticker/fold comparisons per estimator, the time-block bootstrap interval showed no clear improvement for linear regression (13 uncertain, 5 worse, 0 better) and only 2 improvements, 4 worse results, and 12 uncertain results for linear SVR. These are exploratory comparisons without multiple-comparison correction. The shuffled-sentiment placebo was similar to the true-sentiment feature set, which further weakens the evidence that VADER added useful predictive information here.

## Interpretation and next experiment

This first real-data run demonstrates that the recovered files can be joined and aligned to market sessions without a random split. It does not show predictive lift from generic VADER sentiment. The next valuable experiment is to add point-in-time-safe financial/tweet sentiment features and a newer model as a separate sensitivity analysis, then compare them on the same chronological folds. Models pretrained after the historical period must be marked retrospective because their training may contain later market language. A stronger sentiment-quality evaluation also needs human-labeled tweets; this Kaggle source provides no gold polarity labels.

To reproduce the import and baseline after downloading the source files:

```powershell
python -m sentiment_stock import-original --tweets data/external/original-kaggle/Tweet.csv --mapping data/external/original-kaggle/Company_Tweet.csv --output artifacts/original-kaggle-import
python -m sentiment_stock fetch-prices --tickers AAPL AMZN GOOG GOOGL MSFT TSLA --start 2015-01-01 --end 2020-01-05 --output artifacts/original-kaggle-prices
python -m sentiment_stock run --prices artifacts/original-kaggle-prices/prices.csv --posts artifacts/original-kaggle-import/posts.csv --encoder vader --target return --output artifacts/original-kaggle-vader-run --splits 3
```
