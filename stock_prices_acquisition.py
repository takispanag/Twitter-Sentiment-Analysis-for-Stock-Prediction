import yfinance as yf
import pandas as pd


data = yf.download("TSLA", start="2015-01-01", end="2020-12-31").reset_index()
data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
data.to_csv("data/stock_prices/tesla_stock_price.csv", index=False)

data = yf.download("MSFT", start="2015-01-01", end="2020-12-31").reset_index()
data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
data.to_csv("data/stock_prices/microsoft_stock_price.csv", index=False)

data = yf.download("AAPL", start="2015-01-01", end="2020-12-31").reset_index()
data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
data.to_csv("data/stock_prices/apple_stock_price.csv", index=False)

data = yf.download("GOOG", start="2015-01-01", end="2020-12-31").reset_index()
data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
data.to_csv("data/stock_prices/google_stock_price.csv", index=False)

data = yf.download("AMZN", start="2015-01-01", end="2020-12-31").reset_index()
data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
data.to_csv("data/stock_prices/amazon_stock_price.csv", index=False)