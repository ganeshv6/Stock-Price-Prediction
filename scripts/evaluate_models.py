
# scripts/evaluate_models.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

from src.preprocess import prepare_lstm_data

# -------------------------------
# Utility functions
# -------------------------------

def rmse_mae(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def directional_accuracy(actual, predicted):
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    correct = (actual_dir == pred_dir).sum()
    return (correct / len(actual_dir)) * 100

def detect_trend(predicted_prices):
    if predicted_prices[-1] > predicted_prices[0]:
        return "Upward Trend"
    elif predicted_prices[-1] < predicted_prices[0]:
        return "Downward Trend"
    else:
        return "Sideways Trend"

def load_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def evaluate_short_horizon(csv_path="data/RELIANCE_NS_stock_data.csv", horizon=7):
    df = load_clean(csv_path)

    seq_len = 120
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
        df, feature="Close", seq_len=seq_len, train_ratio=0.8
    )

    lstm = load_model("models/lstm_model.h5", compile=False)

    # Last available sequence
    last_seq = X_test[-1]
    lstm_preds = []

    current_seq = last_seq.copy()
    for _ in range(horizon):
        pred_scaled = lstm.predict(current_seq.reshape(1, seq_len, 1), verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        lstm_preds.append(pred)
        current_seq = np.append(current_seq[1:], pred_scaled).reshape(seq_len, 1)

    actual_lstm = df["Close"].values[-horizon:]

    rmse_l, mae_l = rmse_mae(actual_lstm, lstm_preds)
    dir_acc_l = directional_accuracy(actual_lstm, lstm_preds)
    trend_l = detect_trend(lstm_preds)

    train_size = int(len(df) * 0.8)
    train_ts = df["Close"][:train_size]

    arima = sm.tsa.ARIMA(train_ts, order=(7, 1, 3))
    arima_fit = arima.fit()

    # arima_preds = arima_fit.forecast(steps=horizon).values
    # actual_arima = df["Close"].values[-horizon:]
    arima_preds = []
    actual_arima = []

    for i in range(horizon):
        train_ts = df["Close"][:-horizon+i]
        model = sm.tsa.ARIMA(train_ts, order=(5,1,2)).fit()
        pred = model.forecast(steps=1).iloc[0]
        arima_preds.append(pred)
        actual_arima.append(df["Close"].iloc[-horizon+i])

    rmse_a, mae_a = rmse_mae(actual_arima, arima_preds)
    dir_acc_a = directional_accuracy(actual_arima, arima_preds)
    trend_a = detect_trend(arima_preds)

    print("\n=== SHORT-TERM (7-DAY) MODEL EVALUATION ===\n")

    print("LSTM RESULTS")
    print(f"RMSE (7 days): {rmse_l:.2f}")
    print(f"MAE  (7 days): {mae_l:.2f}")
    print(f"Directional Accuracy: {dir_acc_l:.2f}%")
    print(f"Predicted Trend: {trend_l}\n")

    print("ARIMA RESULTS")
    print(f"RMSE (7 days): {rmse_a:.2f}")
    print(f"MAE  (7 days): {mae_a:.2f}")
    print(f"Directional Accuracy: {dir_acc_a:.2f}%")
    print(f"Predicted Trend: {trend_a}")

if __name__ == "__main__":
    evaluate_short_horizon()
