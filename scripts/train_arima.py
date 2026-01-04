# # scripts/train_arima.py
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from src.preprocess import compute_log_returns

# def train_arima_walk_forward(csv_path, p=5, d=1, q=2):
#     df = pd.read_csv(csv_path, low_memory=False)

#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

#     # 🚀 DROP CORRUPTED ROWS (FIXES YOUR ERROR)
#     df = df.dropna(subset=["Date", "Close"])
#     df = df.sort_values("Date").reset_index(drop=True)

#     df = compute_log_returns(df)
#     returns = df["Log_Return"].values

#     split = int(len(returns) * 0.8)
#     history = returns[:split].tolist()
#     test = returns[split:]

#     preds = []
#     for t in range(len(test)):
#         model = sm.tsa.ARIMA(history, order=(p, d, q))
#         fit = model.fit()
#         yhat = fit.forecast()[0]
#         preds.append(yhat)
#         history.append(test[t])

#     os.makedirs("models", exist_ok=True)
#     np.save("models/arima_preds.npy", preds)
#     np.save("models/arima_actual.npy", test)

#     print("✅ ARIMA walk-forward training completed")

# if __name__ == "__main__":
#     train_arima_walk_forward("data/RELIANCE_NS_stock_data.csv")


import os 
import pandas as pd 
import statsmodels.api as sm 
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    # drop junk header rows (if any) 
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower() or "index" in c.lower():
                df.rename(columns={c: "Date"}, inplace=True) 
                break 
    # find close-like column
    if "Close" not in df.columns:
        for c in df.columns:
            if "close" in c.lower():
                df.rename(columns={c: "Close"}, inplace=True)
                break 
    df = df.dropna(subset=["Date"]).reset_index(drop=True) 
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce") 
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce") 
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True) 
    return df 
def train_and_save_arima(csv_path="data/RELIANCE_NS_stock_data.csv", p=5, d=1, q=2, save_path="models/arima_result.pickle"): 
    os.makedirs("models", exist_ok=True) 
    df = load_and_clean(csv_path) 
    ts = df.set_index("Date")["Close"].astype(float) 
    model = sm.tsa.ARIMA(ts, order=(p,d,q)) 
    res = model.fit() 
    res.save(save_path) # statsmodels results save 
    print("Saved ARIMA result to:", save_path) 
    return save_path 
if __name__ == "__main__": 
    # tune p,d,q as you like or use pmdarima.auto_arima externally 
    train_and_save_arima("data/RELIANCE_NS_stock_data.csv", p=5, d=1, q=2)