import yfinance as yf
import pandas as pd
import config
from utils import create_folder

def stock_prices():
    create_folder("data/new_data/stock_prices")
    for company_name in config.company_names:
        data = yf.download(f"{company_name}", start="2014-12-31", end="2021-12-18").reset_index()
        data["Date"] = pd.to_datetime(data["Date"].dt.strftime('%Y/%m/%d'))
        data.to_csv(f"data/new_data/stock_prices/{company_name}_stock_price.csv", index=False)