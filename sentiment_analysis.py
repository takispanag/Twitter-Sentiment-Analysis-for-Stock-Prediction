import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from multiprocessing import Process, Manager
from utils import sentiment_avg

NUMBER_OF_PROCESSES = 20
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
    df = pd.read_csv(f"data/{company_name}_tweets.csv")

    manager = Manager()
    sentiments = manager.dict()
    threads = [0]*NUMBER_OF_PROCESSES
    start = time.time()
    for process in range(NUMBER_OF_PROCESSES):
        threads[process] = Process(target=wordAnalyser, args=(df.iloc[process*15000:(process+1)*15000], sentiments, process,))
        threads[process].start()

    for process in range(NUMBER_OF_PROCESSES):
        threads[process].join()

    end = time.time()
    temp_sentiments = []
    for process in range(NUMBER_OF_PROCESSES):
        temp_sentiments.append(sentiments[process])
    df["sentiment"] = pd.DataFrame(temp_sentiments).to_numpy().flatten()
    # drop rows where sentiment is 0 (neutral)
    df = df[df.sentiment != 0]
    df.iloc[-200000:].to_csv(f"data/stock_sentiment/{company_name}_sentiment.csv", index=False) # get last 200000 rows

    print("Time elapsed = "+(end - start))

if __name__ == "__main__":
    for company_name in ["amazon", "apple", "google", "microsoft", "tesla"]: #["amazon", "apple", "google", "microsoft", "tesla"]
        sentiment_Analysis_Parallel(company_name) #sentiment analysis and write to csv
        sentiment_avg(company_name) #get day avg sentiment and write to csv Format: ["Date", "Sentiment"]