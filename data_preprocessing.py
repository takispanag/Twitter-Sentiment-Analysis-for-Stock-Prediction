import pandas as pd
from utils import cleanText, convertEpochToDate, create_folder


def inference_featurization(company_name):
    df1 = pd.read_csv(f"data/new_data/stock_daily_avg_sentiment/{company_name}_avg_sentiment.csv").set_index("Date")
    df2 = pd.read_csv(f"data/new_data/stock_prices/{company_name}_stock_price.csv").set_index("Date")
    df1["Close"] = df2["Close"]
    return df1.dropna()


def featurization(company_name):
    create_folder('data/training_data')
    df1 = pd.read_csv(f'data/stock_prices/{company_name}_stock_price.csv')
    df2 = pd.read_csv(f'data/stock_daily_avg_sentiment/{company_name}_avg_sentiment.csv')
    df1 = df1.set_index("Date").dropna()
    df2 = df2.set_index("Date").dropna()
    df2["Close"] = df1["Close"]  # Get previous day close
    df2["Next_Close"] = df1["Close"].shift(-1)  # Get Fridays, Saturdays and Holidays
    df2 = df2.dropna()  # Drop Fridays, Saturdays and Holidays
    df2["Close"] = df1["Close"]
    df2.to_csv(f'data/training_data/{company_name}_data.csv')
    return df2


def merge_tables_get_last_tweets(tweets_num):
    tweets = pd.read_csv("data/OriginalDataset/Tweet.csv", index_col=False)
    ids = pd.read_csv("data/OriginalDataset/Company_Tweet.csv", index_col=False)

    df = pd.merge(tweets, ids, on="tweet_id").dropna()  # merge the csv's on tweet_id

    df["body"] = df["body"].apply(cleanText)
    df["post_date"] = df["post_date"].apply(convertEpochToDate)  # convert to date
    create_folder('data/tweets_by_stock')

    # get last 300.000 tweets for each company
    for company_name in ["AMZN", "AAPL", "GOOG", "MSFT", "TSLA"]:
        df.loc[df.ticker_symbol == company_name].dropna().iloc[-tweets_num:].to_csv(
            f"data/tweets_by_stock/{company_name}_tweets.csv", index=False)
