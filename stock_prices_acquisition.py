"""Optional legacy Yahoo price acquisition."""
import yfinance as yf
import pandas as pd
from utils import create_folder


def stock_prices():
    try:
        import config
    except ImportError as exc:
        raise RuntimeError("Legacy price acquisition needs a local config.py with company_names.") from exc
    create_folder("data/new_data/stock_prices")
    for company_name in getattr(config, "company_names", None) or getattr(config, "COMPANIES", []):
        data = yf.download(company_name, start="2014-12-31", end="2021-12-18", auto_adjust=True).reset_index()
        data["Date"] = pd.to_datetime(data["Date"].dt.strftime("%Y/%m/%d"))
        data.to_csv(f"data/new_data/stock_prices/{company_name}_stock_price.csv", index=False)
