import pandas as pd
import re
import datetime as dt
from utils import cleanText, convertEpochToDate


tweets = pd.read_csv("data/OriginalDataset/Tweet.csv", index_col=False)
ids = pd.read_csv("data/OriginalDataset/Company_Tweet.csv", index_col=False)

df = pd.merge(tweets, ids, on="tweet_id").dropna()  # merge the csv's on tweet_id


df["body"] = df["body"].apply(cleanText)
df["post_date"] = df["post_date"].apply(convertEpochToDate) #convert to date
# get last 300.000 tweets for each company
df.loc[df.ticker_symbol == "AAPL"].dropna().iloc[-300000:].to_csv("data/apple_tweets.csv", index=False)
df.loc[df.ticker_symbol == "TSLA"].dropna().iloc[-300000:].to_csv("data/tesla_tweets.csv", index=False)
df.loc[df.ticker_symbol == "AMZN"].dropna().iloc[-300000:].to_csv("data/amazon_tweets.csv", index=False)
df.loc[df.ticker_symbol == "GOOG"].dropna().iloc[-300000:].to_csv("data/google_tweets.csv", index=False)
df.loc[df.ticker_symbol == "MSFT"].dropna().iloc[-300000:].to_csv("data/microsoft_tweets.csv", index=False)