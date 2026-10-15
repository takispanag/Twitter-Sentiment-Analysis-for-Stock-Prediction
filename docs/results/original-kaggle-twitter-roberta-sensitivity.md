# Original Kaggle tweets: Twitter-RoBERTa GPU sensitivity run

This follow-up compares the existing VADER baseline with pinned Cardiff NLP Twitter-RoBERTa scores on the same sampled posts, market panel, and walk-forward folds. It asks whether the newer tweet-language encoder changes next-session return forecasts; it is not a test of classifier accuracy because this Kaggle corpus has no human sentiment labels.

## Data and protocol

The full Kaggle import contains 4,336,445 tweet-to-company rows across AAPL, AMZN, GOOG, GOOGL, MSFT, and TSLA. Transformer inference used a deterministic 36,000-post sample: 100 posts from each of 360 ticker-month groups (six tickers, 60 months). The sample spans 2015-01-01 through 2019-12-31 and retains almost all ticker-session coverage. Both encoders were run against the same 7,560-session adjusted-price panel and the same three expanding chronological folds per ticker. Each model/feature-set metric averages 18 ticker-fold results, with 311 held-out sessions per fold and ticker.

Tweets through each session's close form features for a forecast issued 15 minutes after close. The target is the next NYSE session's open-to-close return. The experiment includes a training-mean naive baseline, price-only features, and price plus sentiment. Runs also include attention, shuffled-sentiment placebos, linear regression, linear SVR, XGBoost, and an LSTM. The reported mean is equal-weighted across ticker-fold rows; lower MAE/RMSE is better.

The classifier used the pinned revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, batch size 32, max length 128, and the NVIDIA RTX 4070 Ti (`cuda:0`). The model card says its base model was pretrained on about 124 million tweets from January 2018 to December 2021, then fine-tuned with TweetEval. That training period extends after the 2015–2019 observations, so this is a retrospective sensitivity analysis and cannot represent a model available at each historical forecast date. See the [official model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

## Results

| Forecast model | Feature set | VADER MAE | Twitter-RoBERTa MAE | VADER direction | Twitter-RoBERTa direction |
|---|---|---:|---:|---:|---:|
| Naive training mean | Naive | 0.009689 | 0.009689 | 0.5064 | 0.5064 |
| Linear regression | Price only | 0.009763 | 0.009763 | 0.5105 | 0.5105 |
| Linear regression | Price + sentiment | 0.009936 | 0.009940 | 0.5079 | 0.5145 |
| Linear SVR | Price only | 0.009916 | 0.009916 | 0.5080 | 0.5080 |
| Linear SVR | Price + sentiment | 0.010258 | 0.010215 | 0.5018 | 0.4873 |
| XGBoost | Price only | 0.010005 | 0.010005 | 0.5066 | 0.5066 |
| XGBoost | Price + sentiment | 0.009964 | 0.010093 | 0.5152 | 0.4950 |
| LSTM | Price only | 0.010628 | 0.010628 | 0.4939 | 0.4939 |
| LSTM | Price + sentiment | 0.012030 | 0.012691 | 0.5129 | 0.5063 |

Adding Twitter-RoBERTa sentiment raised MAE versus price-only for all four model families: by about 1.8% for linear regression, 3.0% for linear SVR, 0.9% for XGBoost, and 19.4% for the LSTM. VADER showed a small XGBoost MAE reduction of about 0.4%, but its result was not consistent across the other models. Directional accuracy moved differently from MAE and did not show a stable win. The shuffled-sentiment placebo further cautions against interpreting any isolated improvement as a dependable sentiment effect.

![Relative MAE change from price-only features for each encoder](../../../../outputs/ORIGINAL_KAGGLE_SENTIMENT_GPU_EXPERIMENT.png)

## Interpretation and limits

This result does not show that a more capable tweet sentiment encoder improves stock-return prediction on this dataset. It does show that the full sampling, model inference, causal feature aggregation, walk-forward evaluation, and model comparison workflow now runs with GPU acceleration. The price-only results match exactly between encoder runs, confirming the folds and non-sentiment baseline are aligned.

The sample is much smaller than the full Kaggle archive, there are no gold sentiment labels, and the Twitter-RoBERTa pretraining period overlaps future-to-the-past information relative to the historical targets. These results are research evidence only, not a trading-performance claim. The next strongest experiment is a blinded, human-labeled subset of these finance tweets to measure sentiment quality directly before investing in fine-tuning or more complex return models.

The raw archive, price CSVs, sampled posts, and row-level run outputs are excluded from Git. Local output folders are `artifacts/original-kaggle-twitterroberta-run/` and `artifacts/original-kaggle-vader-sampled-run/`.
