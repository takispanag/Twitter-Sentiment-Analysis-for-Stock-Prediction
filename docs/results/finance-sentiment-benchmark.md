# Finance sentiment source-transfer benchmark

This experiment classifies published finance-text sentiment labels. It does not test stock-return prediction or trading performance.

## Dataset and split

The locally downloaded [TimKoornstra/financial-tweets-sentiment compilation](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment) reports 38,091 rows from nine source datasets. Its card encodes `0=neutral`, `1=bullish`, `2=bearish`; the importer applies that mapping explicitly. The audit found one blank row, 63 duplicate normalized-text rows, and 38,027 distinct normalized texts after blank removal. The corpus has no timestamp or ticker columns and may mix tweets, headlines, and other finance text. Its MIT card says it follows the original dataset licenses; constituent terms remain unverified. Labels are published source labels, not a uniform expert-annotated gold set.

The entire TFNS constituent source was held out for the final test. Duplicate normalized text that appeared across this test source and the other sources was removed from training. The test has 11,924 examples (1,788 bearish, 7,739 neutral, 2,397 bullish). The other sources were divided by source group into 17,445 training and 8,658 validation examples. No model was selected against test metrics.

## Held-out test results

All models below are evaluated on the same source-held-out examples. Lower Brier and expected calibration error (ECE) are better. The majority and VADER baselines use hard predictions represented as one-hot probabilities, so their calibration scores are overconfident by construction.

| Model | Accuracy | Macro F1 | Balanced accuracy | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Majority | 0.201 | 0.112 | 0.333 | 1.598 | 0.799 |
| VADER | 0.484 | 0.440 | 0.475 | 1.033 | 0.516 |
| Word/character TF-IDF + balanced logistic regression | 0.458 | 0.461 | 0.602 | 0.668 | 0.120 |
| FinBERT frozen | **0.715** | **0.663** | **0.704** | **0.417** | 0.109 |
| Twitter-RoBERTa frozen | 0.692 | 0.593 | 0.575 | 0.430 | **0.076** |
| FinBERT fine-tuned, full source train, 1 epoch | 0.547 | 0.548 | 0.696 | 0.665 | 0.249 |

The frozen FinBERT checkpoint was stronger overall than the fine-tuned variant on this held-out source. Fine-tuning improved bearish recall to 0.909 but shifted errors toward neutral recall (0.388), lowering macro F1. That is evidence of source and label-convention mismatch in the mixed compilation; the result supports keeping the frozen model for this test, not treating a newly fine-tuned encoder as automatically better.

| Model | Bearish F1 | Neutral F1 | Bullish F1 |
|---|---:|---:|---:|
| TF-IDF + balanced logistic regression | 0.473 | 0.442 | 0.468 |
| FinBERT frozen | 0.607 | 0.789 | 0.594 |
| Twitter-RoBERTa frozen | 0.485 | 0.781 | 0.513 |
| FinBERT fine-tuned | 0.497 | 0.546 | 0.600 |

The frozen FinBERT weights were evaluated at revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`; Twitter-RoBERTa at `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`. The fine-tune used FinBERT revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, class-weighted loss, 24-row batches, a learning rate of `2e-5`, maximum sequence length 128, and a fixed seed of 42. The full model checkpoint and input rows are local-only.

## Limits and next data step

The test measures transfer to one constituent's published labels, not human expert accuracy. Frozen-model pretraining overlap cannot be ruled out. The FinTwitBERT sentiment classifier is excluded because its model card reports training on this exact 38,091-row compilation and a synthetic derivative, which overlaps our test source. ProsusAI FinBERT weights are under CC BY-NC-SA 4.0, which includes non-commercial restrictions.

This dataset cannot test stock prediction because it lacks dates, tickers, and return labels. The project's historical price research remains a separate analysis. For a point-in-time experiment, download the [2015–2020 Kaggle `Tweet.csv` and `Company_Tweet.csv` corpus](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) or the [2021–2022 Equinxx tweet/price corpus](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction), put the files under `data/external/original-kaggle/`, and run the documented import command. Neither source supplies expert sentiment labels; sentiment scores from a model remain predictions.
