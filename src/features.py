# src/features.py
import pandas as pd

def add_technical_indicators(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    # SMA/EMA
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    # Returns
    df["Return_1"] = df["Close"].pct_change(1)
    if "Volume" in df.columns:
        df["Vol_Change"] = df["Volume"].pct_change(1)
    df = df.dropna().reset_index(drop=True)
    return df
