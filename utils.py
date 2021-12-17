import pandas as pd
import re
import datetime as dt


def percentage(part, whole):
  return round((float(part)/float(whole)),3)

def sentiment_avg(company):
    df = pd.read_csv(f"data/{company}_sentiment.csv")
    pos_sentiment = 0
    neg_sentiment = 0
    data = []
    for date in df["post_date"].unique():
        pos_sentiment = len(df.loc[(df.post_date == date) & (df.sentiment == 1)].index)
        neg_sentiment = len(df.loc[(df.post_date == date) & (df.sentiment == -1)].index)
        print(f"Date = {date}, Pos = {pos_sentiment}, Neg = {neg_sentiment}")
        print("Positive percentage = " + str(percentage(pos_sentiment, (pos_sentiment + neg_sentiment))))
        data.append([date, str(percentage(pos_sentiment, (pos_sentiment + neg_sentiment)))])
    new_df = pd.DataFrame(data, columns=['Date', 'Sentiment'])
    new_df.to_csv(f"data/stock_avg_sentiment/{company}_avg_sentiment.csv", index=False)

def convertEpochToDate(epoch):
    return dt.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d")

def cleanText(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Remove @mentions
    text = re.sub('#', '', text)  # Remove '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Remove RT (retweets tag)
    text = re.sub('https?:\/\/\S+', '', text)  # Remove links
    text = re.sub('/^\s*$/', '', text)  # Remove empty tweets or tweets with only space
    text = re.sub('^\d+$', '', text)  # Remove tweets containing only numbers
    return text

def missing_dates(company):
    df = pd.read_csv(f"data/{company}_stock_prices.csv")
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    # dates which are not in the sequence
    # are returned
    missing_dates_list = pd.date_range(start="2015-12-31", end="2019-12-30").difference(df.index)
    # print(df_copy.loc[df_copy.Date == "2014-12-31"])  # missing_dates[0] - timedelta(days=1))
    return missing_dates_list