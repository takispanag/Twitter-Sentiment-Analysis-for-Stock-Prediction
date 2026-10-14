import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from multiprocessing import Process, Manager

import config
from utils import percentage, create_folder

NUMBER_OF_PROCESSES = 10


def wordAnalyser(df, sentiment, threadNum):
    sents = []

    for tweet in df["body"]:
        analyzer = SentimentIntensityAnalyzer().polarity_scores(str(tweet))
        neg = analyzer['neg']
        pos = analyzer['pos']
        if neg > pos:
            sents.append(-1)
        elif pos > neg:
            sents.append(1)
        elif pos == neg:
            sents.append(0)
    sentiment[threadNum] = sents
    print(f"Process {threadNum}: finished")


def sentiment_Analysis_Parallel(company_name):
    df = pd.read_csv(f"data/tweets_by_stock/{company_name}_tweets.csv")
    df = df[["post_date", "body"]]
    manager = Manager()
    sentiments = manager.dict()
    threads = [0] * NUMBER_OF_PROCESSES
    arr = []
    for i in range(NUMBER_OF_PROCESSES-1):
        arr.append(df.iloc[i*(len(df)//NUMBER_OF_PROCESSES):(i + 1) * int(len(df) / NUMBER_OF_PROCESSES)])
    arr.append(df.iloc[(i+1)*(len(df)//NUMBER_OF_PROCESSES):])
    for process in range(NUMBER_OF_PROCESSES):
        threads[process] = Process(target=wordAnalyser,
                                   args=(arr[process],
                                         sentiments, process,))
        threads[process].start()

    for process in range(NUMBER_OF_PROCESSES):
        threads[process].join()

    temp_sentiments = np.zeros(0)
    for process in range(NUMBER_OF_PROCESSES):
        temp_sentiments = np.append(temp_sentiments, np.array(sentiments[process]))
    df["sentiment"] = pd.DataFrame(temp_sentiments).to_numpy().flatten()
    # drop rows where sentiment is 0 (neutral)
    df = df[df.sentiment != 0]
    return df.reset_index(drop=True)


def sentiment_avg(company, df):
    # df = pd.read_csv(f"data/new_data/{company}_sentiment.csv")
    pos_sentiment = 0
    neg_sentiment = 0
    data = []
    for date in df["post_date"].unique():
        # Get all rows of positive sentiment (1) for specific date
        pos_sentiment = len(df.loc[(df.post_date == date) & (df.sentiment == 1)].index)
        # Get all rows of negative sentiment (-1) for specific date
        neg_sentiment = len(df.loc[(df.post_date == date) & (df.sentiment == -1)].index)
        # print(f"Date = {date}, Pos = {pos_sentiment}, Neg = {neg_sentiment}")
        # print("Positive percentage = " + str(percentage(pos_sentiment, (pos_sentiment + neg_sentiment))))
        data.append([date, str(percentage(pos_sentiment, (pos_sentiment + neg_sentiment)))])
    new_df = pd.DataFrame(data, columns=['Date', 'Sentiment'])
    create_folder("data/new_data/stock_daily_avg_sentiment/v2")
    new_df.to_csv(f"data/new_data/stock_daily_avg_sentiment/v2/{company}_avg_sentiment.csv", index=False)

def get_historical_sentiment():
    for company_name in config.company_names:
        sentiment_avg(company_name, sentiment_Analysis_Parallel(company_name))  # sentiment analysis and write to csv
        # get day avg sentiment and write to csv Format: ["Date", "Sentiment"]
