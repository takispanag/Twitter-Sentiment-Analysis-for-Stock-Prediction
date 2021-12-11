import yfinance as yf

data = yf.download("TSLA", start="2015-01-01", end="2020-12-31")
data.to_csv("data/tesla_stock_prices.csv")

data = yf.download("MSFT", start="2015-01-01", end="2020-12-31")
data.to_csv("data/microsoft_stock_prices.csv")

data = yf.download("AAPL", start="2015-01-01", end="2020-12-31")
data.to_csv("data/apple_stock_prices.csv")

data = yf.download("GOOG", start="2015-01-01", end="2020-12-31")
data.to_csv("data/google_stock_prices.csv")

data = yf.download("AMZN", start="2015-01-01", end="2020-12-31")
data.to_csv("data/amazon_stock_prices.csv")