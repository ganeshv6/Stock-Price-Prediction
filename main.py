from src.data_loader import load_stock_data
from src.preprocess import prepare_lstm_data
from src.lstm_model import build_lstm_model
from src.arima_model import build_arima
from src.evaluate import evaluate
import pandas as pd
import numpy as np

df = load_stock_data("AAPL")
df = df[["Close"]]

# Prepare data for LSTM
X, y, scaler = prepare_lstm_data(df)
train_len = int(len(X) * 0.80)
X_train, X_test = X[:train_len], X[train_len:]
y_train, y_test = y[:train_len], y[train_len:]

# LSTM model
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# LSTM predictions
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
actual = scaler.inverse_transform(y_test)

print("LSTM Evaluation:", evaluate(actual, pred))

# ARIMA model
arima = build_arima(df)
print(arima.summary())
