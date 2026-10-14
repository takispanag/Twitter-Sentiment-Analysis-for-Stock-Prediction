"""Optional legacy collector. It requires local credentials only when explicitly run."""
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from utils import cleanText
from sentiment_analysis import sentiment_Analysis_Parallel, sentiment_avg


def get_tweets():
    try:
        import config
        import tweepy
        from config import consumer_key, consumer_secret, access_token, access_token_secret
    except ImportError as exc:
        raise RuntimeError("Legacy collection needs config.py and tweepy credentials; prefer the explicit collect-x CLI.") from exc
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    Path("data/new_data").mkdir(parents=True, exist_ok=True)
    for company_name in getattr(config, "company_names", None) or getattr(config, "COMPANIES", []):
        rows = []
        start_date, end_date = date(2021, 12, 27), date(2022, 1, 3)
        while start_date <= end_date:
            tweets = tweepy.Cursor(api.search_tweets, q=f"#{company_name} OR {company_name} OR {company_name.lower()}", count=500, lang="en", until=start_date, tweet_mode="extended").items(500)
            rows.extend([[tweet.created_at.date(), tweet.full_text] for tweet in tweets])
            start_date += timedelta(days=1)
        raw = pd.DataFrame(rows, columns=["post_date", "body"])
        if not raw.empty:
            raw["body"] = raw.body.map(cleanText)
        raw.to_csv(f"data/new_data/{company_name}_new_data.csv", index=False)
        scored = sentiment_Analysis_Parallel(company_name, raw)
        sentiment_avg(company_name, scored)

if __name__ == "__main__":
    get_tweets()
