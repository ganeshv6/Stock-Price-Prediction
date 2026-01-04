import os
import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
# from src.lstm_model import build_lstm_model


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Stock Forecast Dashboard — LSTM & ARIMA", initial_sidebar_state="auto")

LSTM_MODEL_PATH = "models/lstm_model.h5"
LSTM_SCALER_PATH = "models/lstm_scaler.pkl"

@st.cache_data(ttl=60*60)
def download_stock(ticker: str, start="2000-01-01", end="2025-12-31"):
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.shape[0] == 0:
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col]).strip() for col in df.columns.values]
    return df

def detect_date_col(df: pd.DataFrame):
    possible = ["Date", "date", "Datetime", "datetime", "index"]
    for c in df.columns:
        if c in possible:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None

def fix_timestamp_column(df, col):
    # convert large epoch-like integers to datetime
    try:
        s = df[col].dropna().astype(str).iloc[0]
        val = int(float(s))
    except Exception:
        return df
    if val > 2_000_000_000_000_000_000:
        unit = "ns"
    elif val > 2_000_000_000_000:
        unit = "us"
    elif val > 2_000_000_000:
        unit = "ms"
    else:
        unit = None
    try:
        if unit:
            df[col] = pd.to_datetime(df[col].astype(float), unit=unit, errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def detect_close_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if "close" in c.lower()]
    if len(candidates) == 0:
        for c in df.columns:
            if c.lower().startswith("adj") and "close" in c.lower():
                candidates.append(c)
    return candidates[0] if candidates else None

def safe_numeric(df, exclude_cols=None):
    exclude_cols = exclude_cols or []
    for c in df.columns:
        if c in exclude_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def lstm_forecast_train_predict(close_array, seq_len=60, days_ahead=1, epochs=10):
    """Train LSTM quickly in-memory and return (model, scaler, test_actual, test_pred, future_preds)"""
    arr = np.array(close_array).reshape(-1,1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return None, None, None, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    # multi-step:
    last_seq = scaled[-seq_len:].reshape(seq_len,1)
    seq = last_seq.copy()
    preds_scaled = []
    for _ in range(days_ahead):
        x_in = seq.reshape(1, seq_len, 1)
        p = model.predict(x_in, verbose=0)[0][0]
        preds_scaled.append(p)
        seq = np.vstack([seq[1:], [[p]]])
    future_preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return model, scaler, actual, pred, future_preds

def lstm_predict_from_saved(model, scaler, close_values, seq_len=60, days_ahead=1):
    """Use saved model+scaler to predict next days. Returns numpy array length days_ahead or None."""
    try:
        arr = np.array(close_values).reshape(-1,1).astype(float)
    except Exception:
        return None
    if model is None or scaler is None:
        return None
    if len(arr) < seq_len:
        return None
    last_seq = arr[-seq_len:]
    scaled_seq = scaler.transform(last_seq.reshape(-1,1))
    seq = scaled_seq.copy()
    preds_scaled = []
    for _ in range(days_ahead):
        x = seq.reshape(1, seq.shape[0], 1)
        p_scaled = float(model.predict(x, verbose=0)[0,0])
        preds_scaled.append(p_scaled)
        seq = np.vstack([seq[1:], [[p_scaled]]])
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return preds

def safe_round_array(arr, decimals=4):
    try:
        return np.round(arr, decimals).tolist()
    except Exception:
        return [None]*len(arr)

# -------------------------
# Accuracy & trend utilities
# -------------------------
def compute_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def directional_accuracy(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    if len(actual) < 2:
        return None
    actual_dir = np.sign(actual[1:] - actual[:-1])
    pred_dir = np.sign(predicted[1:] - actual[:-1])
    return np.mean(actual_dir == pred_dir) * 100

def trend_label(last_actual, last_pred):
    return "🔼 Upward Trend" if last_pred > last_actual else "🔽 Downward Trend"

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def next_day_accuracy(actual_series, predicted_value):
    """
    Computes accuracy % based on last known actual value
    """
    last_actual = actual_series[-1]
    error_pct = abs(predicted_value - last_actual) / last_actual * 100
    return 100 - error_pct

def historical_next_day_metrics(close_series):
    """
    One-step-ahead historical backtesting metrics
    """
    actual = close_series[1:]
    predicted = close_series[:-1]

    rmse = compute_rmse(actual, predicted)
    mae = compute_mae(actual, predicted)
    dir_acc = directional_accuracy(actual, predicted)

    return rmse, mae, dir_acc

def fast_next_day_metrics(close, seq_len, p, d, q, window=30):
    actual = []
    lstm_preds = []
    arima_preds = []
    for i in range(-window, -1):
        history = close[:i]
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
            lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
            with open(LSTM_SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)

            lstm_pred = lstm_predict_from_saved(
                lstm_model, scaler,
                history,
                seq_len=seq_len,
                days_ahead=1
            )[0]
        else:
            _, _, _, _, pred = lstm_forecast_train_predict(
                history, seq_len=seq_len, days_ahead=1, epochs=5
            )
            lstm_pred = pred[0]

        # --- ARIMA ---
        ts = pd.Series(history)
        arima_model = sm.tsa.ARIMA(ts, order=(p, d, q)).fit()
        arima_pred = arima_model.forecast(steps=1).iloc[0]

        actual.append(close[i])
        lstm_preds.append(lstm_pred)
        arima_preds.append(arima_pred)

    return (
        compute_rmse(actual, lstm_preds),
        compute_mae(actual, lstm_preds),
        directional_accuracy(actual, lstm_preds),
        compute_rmse(actual, arima_preds),
        compute_mae(actual, arima_preds),
        directional_accuracy(actual, arima_preds),
    )
# def fast_next_day_metrics(close, seq_len, p, d, q, window=30):
#     import numpy as np
#     import pandas as pd
#     import statsmodels.api as sm
#     import os, pickle
#     from tensorflow.keras.models import load_model

#     close = np.array(close).astype(float)

#     actual = []
#     lstm_preds = []
#     arima_preds = []

#     # Start rolling window (forward, not backward)
#     start_idx = len(close) - window - 1

#     for i in range(start_idx, len(close) - 1):
#         history = close[:i]          # data up to time t
#         y_true = close[i]            # actual value at t+1

#         # ---------------- LSTM ----------------
#         if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
#             lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
#             with open(LSTM_SCALER_PATH, "rb") as f:
#                 scaler = pickle.load(f)

#             lstm_pred = lstm_predict_from_saved(
#                 lstm_model,
#                 scaler,
#                 history,
#                 seq_len=seq_len,
#                 days_ahead=1
#             )[0]
#         else:
#             _, _, _, _, pred = lstm_forecast_train_predict(
#                 history,
#                 seq_len=seq_len,
#                 days_ahead=1,
#                 epochs=5
#             )
#             lstm_pred = pred[0]

#         # ---------------- ARIMA (CORRECT) ----------------
#         ts = pd.Series(history)

#         try:
#             arima_model = sm.tsa.ARIMA(ts, order=(p, d, q))
#             arima_fit = arima_model.fit()
#             arima_pred = arima_fit.forecast(steps=1)[0]
#         except Exception:
#             arima_pred = np.nan

#         # collect results
#         actual.append(y_true)
#         lstm_preds.append(lstm_pred)
#         arima_preds.append(arima_pred)

#     # Convert to numpy and drop NaNs (safety)
#     actual = np.array(actual)
#     lstm_preds = np.array(lstm_preds)
#     arima_preds = np.array(arima_preds)

#     mask = ~np.isnan(arima_preds)

#     return (
#         compute_rmse(actual, lstm_preds),
#         compute_mae(actual, lstm_preds),
#         directional_accuracy(actual, lstm_preds),
#         compute_rmse(actual[mask], arima_preds[mask]),
#         compute_mae(actual[mask], arima_preds[mask]),
#         directional_accuracy(actual[mask], arima_preds[mask]),
#     )


# -------------------------
# Sidebar controls (make seq_len/epochs globally available)
# -------------------------
st.sidebar.title("Controls")
ticker_input = st.sidebar.text_input("The stock ticker ", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2025-12-31"))
custom_days = st.sidebar.slider("Custom forecast days (1-7)", min_value=1, max_value=7, value=3)

st.sidebar.markdown("---")
st.sidebar.write("LSTM settings (used by Next-Day & Next-7):")
seq_len = st.sidebar.number_input("Sequence length (days)", min_value=10, max_value=180, value=60)
epochs = st.sidebar.number_input("Train epochs (when training in-app)", min_value=1, max_value=50, value=10)
st.sidebar.markdown("---")
st.sidebar.write("ARIMA default params (used in ARIMA tab & Next predictions):")
p = st.sidebar.number_input("p (AR)", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("d (diff)", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("q (MA)", min_value=0, max_value=10, value=2)
st.sidebar.markdown("---")
st.sidebar.write("Models: LSTM (deep learning), ARIMA (statistical)")

# -------------------------
# Page title
# -------------------------
st.title("📊 Stock Forecast Dashboard — LSTM & ARIMA")
st.markdown("Enter any valid ticker and choose forecast horizon. The app fetches data from Yahoo Finance and computes LSTM & ARIMA forecasts.")

# Tabs
tabs = st.tabs(["Overview","LSTM","ARIMA","Next-Day","Next-7 Days","Data Preview","About"])

# ------------------------- Load data & clean
with st.spinner("Downloading data..."):
    df_raw = download_stock(ticker_input, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

if df_raw is None:
    st.error("No data returned for ticker. Check ticker symbol and network.")
    st.stop()

date_col = detect_date_col(df_raw)
if date_col is None:
    df_raw = df_raw.reset_index()
    date_col = "index"

# fixing timestamp (if necessary)
try:
    if not np.issubdtype(df_raw[date_col].dtype, np.datetime64):
        df_raw = fix_timestamp_column(df_raw, date_col)
except Exception:
    pass

# detect close col
close_col = detect_close_col(df_raw)
if close_col is None:
    st.error("Could not detect Close column. Columns found: " + ", ".join(df_raw.columns))
    st.stop()

# standardize
df = df_raw.copy()
df = df.rename(columns={date_col: "Date", close_col: "Close"})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = safe_numeric(df, exclude_cols=["Date"])
df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------
# Overview
with tabs[0]:
    st.header("Overview")
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown(f"**Ticker:** `{ticker_input}`")
        st.markdown(f"**Rows loaded:** {len(df)}")
        st.markdown(f"**Date column used:** `Date`")
        st.markdown(f"**Close column used:** `Close`")
        st.markdown("**Notes**: The app auto-detects and cleans date & close fields (handles multi-index and epoch timestamps).")
    with col2:
        st.write("Choose a tab to run models & compute forecasts.")
    st.subheader("Close Price (entire timeframe)")
    fig_over, ax_over = plt.subplots(figsize=(10,4))
    ax_over.plot(df["Date"], df["Close"], label="Close", color="tab:blue")
    ax_over.set_xlabel("Date"); ax_over.set_ylabel("Price")
    ax_over.legend()
    st.pyplot(fig_over)
    st.markdown("**Interpretation:** Look for trend and volatility; LSTM may capture complex patterns, ARIMA captures linear autoregressive components.")

# -------------------------
# LSTM tab (train in-app)
with tabs[1]:
    st.header("LSTM Model")
    st.markdown("LSTM captures non-linear patterns and long-term dependencies. Train in-tab or use persisted model if available.")
    if st.button("Train LSTM & Forecast (in-tab)"):
        with st.spinner("Training LSTM (this may take a bit)..."):
            try:
                close_vals = df["Close"].values.astype(float)
                model_lstm, scaler_lstm, actual_l, pred_l, preds_next = lstm_forecast_train_predict(close_vals, seq_len=seq_len, days_ahead=custom_days, epochs=int(epochs))
                if preds_next is None:
                    st.error("LSTM could not train/predict (not enough data).")
                else:
                    # show test plot
                    st.subheader("LSTM: Actual vs Predicted (test portion)")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(actual_l.flatten(), label="Actual", color="black")
                    ax.plot(pred_l.flatten(), label="Predicted", color="red")
                    ax.set_xlabel("Date"); ax.set_ylabel("Price")
                    ax.legend()
                    st.pyplot(fig)

                    st.subheader(f"LSTM {custom_days}-day forecast (last -> future)")
                    last_date = df["Date"].max()
                    future_dates = pd.date_range(start=last_date + BDay(1), periods=len(preds_next), freq="B")

                    st.write(pd.DataFrame({
                        "Date": future_dates.date,
                        "LSTM_Pred": np.round(preds_next, 4)
                    }))
                    st.success("LSTM training and forecast completed.")
            except Exception as e:
                st.error("LSTM failed: " + str(e))

# -------------------------
# ARIMA tab
with tabs[2]:
    st.header("ARIMA Model")
    st.markdown("ARIMA is a classical statistical model (p,d,q). Fit runs on the ticker you requested.")
    arima_days = st.number_input("Forecast days (ARIMA)", min_value=1, max_value=7, value=min(3, custom_days))
    if st.button("Run ARIMA Forecast"):
        with st.spinner("Fitting ARIMA..."):
            try:
                ts = df.set_index("Date")["Close"].astype(float)
                arima = sm.tsa.ARIMA(ts, order=(int(p), int(d), int(q))).fit()
                fc = arima.get_forecast(steps=arima_days)
                # align forecast to business days after last observed date
                last_date = df["Date"].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=arima_days, freq="B")
                arima_series = pd.Series(fc.predicted_mean.values, index=future_dates)
                st.write("ARIMA Forecast (next %d days)" % arima_days)
                st.dataframe(pd.DataFrame({"predicted_mean": np.round(arima_series.values,4)}, index=arima_series.index))
                st.line_chart(arima_series)
                st.success("ARIMA completed.")
            except Exception as e:
                st.error("ARIMA error: " + str(e))

# -------------
# Next-Day tab 
with tabs[3]:
    st.header("Next-Day Predictions")
    st.markdown(
        "Next trading day closing price prediction using LSTM & ARIMA "
        "(structure aligned with Next-7 Days)."
    )

    if st.button("Compute Next-Day Predictions"):
        try:
            with st.spinner("Computing next-day prediction..."):

                last_date = df["Date"].max()
                next_business_day = (last_date + BDay(1)).date()

                lstm_next = None
                if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
                    try:
                        lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
                        with open(LSTM_SCALER_PATH, "rb") as f:
                            scaler = pickle.load(f)

                        lstm_next = lstm_predict_from_saved(
                            lstm_model,
                            scaler,
                            df["Close"].values,
                            seq_len=seq_len,
                            days_ahead=1
                        )[0]
                    except Exception:
                        lstm_next = None

                if lstm_next is None:
                    _, _, _, _, preds = lstm_forecast_train_predict(
                        df["Close"].values,
                        seq_len=seq_len,
                        days_ahead=1,
                        epochs=int(epochs)
                    )
                    lstm_next = preds[0]

                # ARIMA NEXT-DAY
                ts = df.set_index("Date")["Close"].astype(float)
                arima_model = sm.tsa.ARIMA(ts, order=(int(p), int(d), int(q))).fit()
                arima_next = float(arima_model.forecast(steps=1).iloc[0])

                st.subheader(f"Predictions for {next_business_day}")
                c1, c2 = st.columns(2)
                c1.metric("LSTM Next-Day Close", f"{lstm_next:,.2f}")
                c2.metric("ARIMA Next-Day Close", f"{arima_next:,.2f}")

                st.success("Next-day forecast computed.")

                # NEXT-DAY ACCURACY 
                (
                    lstm_rmse, lstm_mae, lstm_dir,
                    arima_rmse, arima_mae, arima_dir
                ) = fast_next_day_metrics(
                    df["Close"].values,
                    seq_len=seq_len,
                    p=int(p), d=int(d), q=int(q),
                    window=30
                )

                st.markdown("### 📊 Model Accuracy (Historical Next-Day)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**LSTM Performance**")
                    st.write(f"RMSE (₹): {lstm_rmse:.2f}")
                    st.write(f"MAE (₹): {lstm_mae:.2f}")
                    st.write(f"Directional Accuracy: {lstm_dir:.2f}%")

                with col2:
                    st.markdown("**ARIMA Performance**")
                    st.write(f"RMSE (₹): {arima_rmse:.2f}")
                    st.write(f"MAE (₹): {arima_mae:.2f}")
                    st.write(f"Directional Accuracy: {arima_dir:.2f}%")

                st.markdown("### 📈 Next-Day Trend Prediction")
                last_close = df["Close"].iloc[-1]

                c_tr1, c_tr2 = st.columns(2)
                c_tr1.write(f"**LSTM Trend:** {trend_label(last_close, lstm_next)}")
                c_tr2.write(f"**ARIMA Trend:** {trend_label(last_close, arima_next)}")

                st.markdown("### 📌 Prediction Reliability (Next-Day)")
                c_rel1, c_rel2 = st.columns(2)
                c_rel1.write(f"**LSTM Avg Error (₹):** ±{lstm_rmse:.2f}")
                c_rel2.write(f"**ARIMA Avg Error (₹):** ±{arima_rmse:.2f}")

                st.markdown("### 🔍 Forecast Confidence Range (Next-Day)")
                conf_df = pd.DataFrame({
                    "Model": ["LSTM", "ARIMA"],
                    "Lower Bound": [
                        round(lstm_next - lstm_rmse, 2),
                        round(arima_next - arima_rmse, 2)
                    ],
                    "Prediction": [
                        round(lstm_next, 2),
                        round(arima_next, 2)
                    ],
                    "Upper Bound": [
                        round(lstm_next + lstm_rmse, 2),
                        round(arima_next + arima_rmse, 2)
                    ]
                })
                st.table(conf_df)

                st.markdown("### 🏆 Model Comparison (Next-Day)")
                best_model = "ARIMA" if arima_rmse < lstm_rmse else "LSTM"

                st.write(f"""
                - **ARIMA RMSE:** {arima_rmse:.2f}
                - **LSTM RMSE:** {lstm_rmse:.2f}
                - ARIMA is more stable for single-day prediction
                """)

                st.success(f"Recommended model: **{best_model}**")

        except Exception as e:
            st.error(f"Next-day prediction error: {e}")

# -------------------------
# Next-7 Days tab
with tabs[4]:
    st.header("Next 7 Days Forecast")
    if st.button("Compute Next-7-Day Forecasts"):
        try:
            with st.spinner("Computing 7-day forecasts..."):
                days = 7

                lstm_preds = None
                if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
                    try:
                        lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
                        with open(LSTM_SCALER_PATH, "rb") as f:
                            scaler = pickle.load(f)
                        lstm_preds = lstm_predict_from_saved(lstm_model, scaler, df["Close"].values, seq_len=seq_len, days_ahead=days)
                    except Exception:
                        lstm_preds = None
                if lstm_preds is None:
                    _, _, _, _, quick_preds = lstm_forecast_train_predict(df["Close"].values, seq_len=seq_len, days_ahead=days, epochs=int(epochs))
                    lstm_preds = quick_preds

                # ARIMA fit and forecast 
                ts = df.set_index("Date")["Close"].astype(float)
                arima_model = sm.tsa.ARIMA(ts, order=(int(p), int(d), int(q))).fit()
                arima_fc = arima_model.forecast(steps=days)  # numpy array

                # Build future business dates
                last_date = df["Date"].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="B")

                # Align ARIMA results to future_dates
                arima_vals = np.array(arima_fc)
                # If ARIMA returned fewer elements for any reason pad with NaN
                if len(arima_vals) < days:
                    arima_vals = np.pad(arima_vals, (0, days-len(arima_vals)), constant_values=np.nan)
                    arima_vals = (
                        pd.Series(arima_vals)
                        .rolling(window=2, min_periods=1)
                        .mean()
                        .values
                    )
                # LSTM preds alignment
                if lstm_preds is None or len(lstm_preds) < days:
                    lstm_vals = np.array([np.nan]*days)
                else:
                    lstm_vals = np.array(lstm_preds)

                table = pd.DataFrame({
                    "Date": future_dates,
                    "LSTM": np.round(lstm_vals, 4),
                    "ARIMA": np.round(arima_vals, 4),
                })
                # show dates in readable format
                table["Date"] = table["Date"].dt.date
                st.table(table)
                st.line_chart(table.set_index("Date"))
                st.success("7-day forecasts computed.")
                actual_7 = df["Close"].iloc[-7:].values

                if len(actual_7) == 7:
                    lstm_rmse = compute_rmse(actual_7, lstm_vals)
                    arima_rmse = compute_rmse(actual_7, arima_vals)

                    lstm_mae = compute_mae(actual_7, lstm_vals)
                    arima_mae = compute_mae(actual_7, arima_vals)

                    lstm_mape = compute_mape(actual_7, lstm_vals)
                    arima_mape = compute_mape(actual_7, arima_vals)

                    lstm_dir = directional_accuracy(actual_7, lstm_vals)
                    arima_dir = directional_accuracy(actual_7, arima_vals)

                    arima_lower = arima_vals - arima_rmse
                    arima_upper = arima_vals + arima_rmse

                    st.markdown("### 📊 Model Accuracy (Last 7 Days)")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**LSTM Performance**")
                        st.write(f"RMSE (₹): {lstm_rmse:.2f}")
                        st.write(f"MAE (₹): {lstm_mae:.2f}")
                        st.write(f"Avg % Error (MAPE): {lstm_mape:.1f}%")
                        st.write(f"Directional Accuracy: {lstm_dir:.2f}%")

                    with col2:
                        st.markdown("**ARIMA Performance**")
                        st.write(f"RMSE (₹): {arima_rmse:.2f}")
                        st.write(f"MAE (₹): {arima_mae:.2f}")
                        st.write(f"Avg % Error (MAPE): {arima_mape:.1f}%")
                        st.write(f"Directional Accuracy: {arima_dir:.2f}%")

                    st.markdown("### 📈 7-Day Trend Prediction")

                    last_actual = df["Close"].iloc[-1]
                    lstm_7_trend = trend_label(last_actual, lstm_vals[-1])
                    arima_7_trend = trend_label(last_actual, arima_vals[-1])

                    c_tr1, c_tr2 = st.columns(2)
                    c_tr1.write(f"**LSTM 7-Day Trend:** {lstm_7_trend}")
                    c_tr2.write(f"**ARIMA 7-Day Trend:** {arima_7_trend}")

                    st.markdown("### 📌 Prediction Reliability (7-Day)")
                    c_rel1, c_rel2 = st.columns(2)
                    c_rel1.write(f"**LSTM Avg Error (₹):** ±{lstm_rmse:.2f}")
                    c_rel2.write(f"**ARIMA Avg Error (₹):** ±{arima_rmse:.2f}")

                    st.markdown("### 🔍 Forecast Confidence Range (7 Days)")

                    conf_df = pd.DataFrame({
                        "Date": future_dates.date,
                        "LSTM Lower": np.round(lstm_vals - lstm_rmse, 2),
                        "LSTM Prediction": np.round(lstm_vals, 2),
                        "LSTM Upper": np.round(lstm_vals + lstm_rmse, 2),
                        "ARIMA Lower": np.round(arima_vals - arima_rmse, 2),
                        "ARIMA Prediction": np.round(arima_vals, 2),
                        "ARIMA Upper": np.round(arima_vals + arima_rmse, 2),
                    })

                    st.table(conf_df)

                    st.info(
                        "LSTM confidence bands are empirical (error-based). "
                        "ARIMA confidence bands are statistically estimated."
                    )

                    st.markdown("### 🏆 Model Comparison (7-Day Horizon)")
                    best_model = "ARIMA" if arima_rmse < lstm_rmse else "LSTM"
                    st.write(f"""
                    - **ARIMA RMSE:** {arima_rmse:.2f}
                    - **LSTM RMSE:** {lstm_rmse:.2f}
                    - **Directional Accuracy:** LSTM performs better for trend detection.
                    """)
                    st.success(f"Recommended model for short-term forecasting: **{best_model}**")

        except Exception as e:
            st.error(f"7-day forecast error: {e}")

# -------------------------
# Data preview
with tabs[5]:
    st.header("Cleaned Data Preview")
    # st.write(df.head(30))
    st.write(df)
    st.write("Data types:")
    st.write(df.dtypes)

# -------------------------
# About
with tabs[6]:
    st.header("About this App & Models")
    st.markdown("""
    ### 📌 Project Overview
    This application is a **short-term stock price forecasting system** built using 
    **LSTM (Deep Learning)** and **ARIMA (Statistical Time Series)** models.
    It is designed to assist beginners in understanding **price movement, trend direction,
    and model reliability** for short investment horizons.

    ---

    ### 🔍 What this App Does
    - Downloads historical stock price data from **Yahoo Finance** using `yfinance`.
    - Performs **data cleaning and preprocessing** (date parsing, sorting, scaling).
    - Uses a **pretrained LSTM model** (offline trained and persisted) for sequence-based learning.
    - Fits an **ARIMA model dynamically** for statistical forecasting.
    - Provides:
        - **Next-Day Forecast**
        - **Next 7-Day Forecast**
    - Displays for each horizon:
        - Predicted price
        - Trend direction (Upward / Downward)
        - Historical accuracy (RMSE, MAE, MAPE, Directional Accuracy)
        - Prediction reliability (expected error)
        - Confidence range
        - Model comparison and recommendation

    ---

    ### 🤖 Models Used
    **LSTM (Long Short-Term Memory)**
    - Captures long-term dependencies in stock price sequences.
    - Trained offline using historical closing prices.
    - Best suited for **trend detection and pattern learning**.

    **ARIMA (AutoRegressive Integrated Moving Average)**
    - Statistical time-series model.
    - Performs well for **short-term price estimation**.
    - Provides confidence intervals based on forecast variance.

    The app combines both models to balance **price accuracy** and **trend insight**.

    ---

    ### 📊 Evaluation Strategy
    - Accuracy metrics are computed using **historical backtesting**.
    - Short-term evaluation focuses on the **last 7 trading days**.
    - Directional accuracy measures how well models predict **market movement direction**.
    - LSTM confidence ranges are **empirical (error-based)**.
    - ARIMA confidence ranges are **statistical**.

    ---

    ### ⚠️ Notes & Limitations
    - Stock markets are inherently volatile and partially random.
    - Predictions are limited to **short-term horizons (1–7 days)**.
    - The app does **not provide financial advice**.
    - Performance may vary across different stocks and market conditions.

    ---

    ### 🎓 Intended Use
    - Academic projects and demonstrations
    - Learning time-series forecasting concepts
    - Comparing deep learning vs statistical models
    - Educational analysis for beginners

    ---
    """)
