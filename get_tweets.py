import tweepy
import config
import pandas as pd
from utils import cleanText
from datetime import date, timedelta
from sentiment_analysis import sentiment_Analysis_Parallel,sentiment_avg
from config import consumer_key,consumer_secret,access_token,access_token_secret


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets():
    for company_name in config.company_names:
        df = pd.DataFrame()
        start_date = date(2021, 12, 27)
        end_date = date(2022, 1, 3)
        delta = timedelta(days=1)
        print(f"Company = {company_name}")
        while start_date <= end_date:
            fetch_tweets = tweepy.Cursor(api.search_tweets,
                                         q=f"#{company_name} OR {company_name} OR {company_name.lower()}", count=500,
                                         lang="en", until=start_date, tweet_mode="extended").items(500)
            df1 = pd.DataFrame(data=[[tweet_info.created_at.date(), tweet_info.full_text] for tweet_info in fetch_tweets],columns=['post_date', 'body'])
            df = df.append(df1)
            start_date += delta
        df['body'] = df['body'].apply(cleanText)

        df.to_csv(f"data/new_data/{company_name}_new_data.csv", index=False)
        sentiment_Analysis_Parallel(company_name)
        sentiment_avg(company_name)


if __name__ == '__main__':
    get_tweets()