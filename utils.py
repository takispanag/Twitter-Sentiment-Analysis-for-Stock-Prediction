import pandas as pd
import re
import datetime as dt
import os

def create_folder(path):
    if os.path.exists(path):
        print("Path Exists")
    else:
        os.mkdir(path)
        print("Path Created")

def percentage(part, whole):
  return round((float(part)/float(whole)),3)



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
    missing_dates_list = pd.date_range(start="2015-01-01", end="2019-12-30").difference(df.index)
    # print(df_copy.loc[df_copy.Date == "2014-12-31"])  # missing_dates[0] - timedelta(days=1))
    return missing_dates_list