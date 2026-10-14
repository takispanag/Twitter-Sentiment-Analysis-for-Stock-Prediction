"""Example legacy settings. Copy to config.py and provide credentials separately."""

COMPANIES = ["amazon", "apple", "google", "microsoft", "tesla"]
TICKER_SYMBOLS = {
    "amazon": "AMZN",
    "apple": "AAPL",
    "google": "GOOG",
    "microsoft": "MSFT",
    "tesla": "TSLA",
}

SENTIMENT_ENCODER = "vader"  # Options: vader, finbert, twitter, provided
FINBERT_MODEL = "ProsusAI/finbert"
TWITTER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Set X_BEARER_TOKEN in the process environment only when running collect-x.
