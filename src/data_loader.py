# import yfinance as yf
# import pandas as pd

# def load_stock_data(ticker="RELIANCE.NS", start="2015-01-01", end="2024-12-31"):
#     data = yf.download(ticker, start=start, end=end)

#     data.reset_index(inplace=True)

#     data.to_csv("data/stock_data.csv", index=False)
#     return data

# if __name__ == "__main__":
#     print(load_stock_data("RELIANCE.NS").head())


# src/data_loader.py
import yfinance as yf
import pandas as pd
import os

def load_stock_data(ticker="RELIANCE.NS", start="2000-01-01", end=None, save_csv=True):
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.shape[0] == 0:
        raise RuntimeError(f"No data returned for ticker {ticker}")
    df = df.reset_index()
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker.replace('.', '_')}_stock_data.csv"
    if save_csv:
        df.to_csv(filename, index=False)
    return df

if __name__ == "__main__":
    print(load_stock_data("RELIANCE.NS", start="2000-01-01").tail())
