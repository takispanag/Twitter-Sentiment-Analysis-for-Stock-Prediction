import pandas as pd

tweets = pd.read_csv("C://Users//Takis//Downloads//Tweet.csv")
ids = pd.read_csv("C://Users//Takis//Downloads//Company_Tweet.csv")

df = pd.merge(tweets, ids, on="tweet_id")

#save each hashtag tweets in new csv file
df.loc[df.ticker_symbol=="AAPL"].to_csv("data/apple_tweets.csv")
df.loc[df.ticker_symbol=="TSLA"].to_csv("data/tesla_tweets.csv")
df.loc[df.ticker_symbol=="AMZN"].to_csv("data/amazon_tweets.csv")
df.loc[df.ticker_symbol=="GOOG"].to_csv("data/google_tweets.csv")
df.loc[df.ticker_symbol=="MSFT"].to_csv("data/microsoft_tweets.csv")