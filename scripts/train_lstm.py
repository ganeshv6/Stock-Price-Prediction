# # scripts/train_lstm.py
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import pandas as pd
# import numpy as np
# import pickle
# from src.preprocess import prepare_lstm_data
# from src.lstm_model import build_lstm_model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# def train_lstm(csv_path="data/RELIANCE_NS_stock_data.csv",
#                seq_len=60, epochs=30, batch_size=32,
#                save_model_path="models/lstm_model.h5",
#                save_scaler_path="models/lstm_scaler.pkl"):

#     os.makedirs("models", exist_ok=True)

#     # -------------------------------
#     # LOAD CSV + REMOVE BAD ROWS
#     # -------------------------------
#     df = pd.read_csv(csv_path)

#     # 🚀 FIX: remove corrupted header-like row
#     df = df.dropna(subset=["Date"]).reset_index(drop=True)

#     # Convert to correct types
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
#     df = df.dropna(subset=["Date","Close"]).reset_index(drop=True)

#     df = df.sort_values("Date").reset_index(drop=True)

#     # -------------------------------
#     # CREATE LSTM DATA
#     # -------------------------------
#     X_train, X_test, y_train, y_test, scaler =prepare_lstm_data(df, feature="Close", seq_len=seq_len, train_ratio=0.8)

#     model = build_lstm_model((X_train.shape[1], 1))

#     mc = ModelCheckpoint(save_model_path, monitor="val_loss", save_best_only=True)
#     es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

#     model.fit(
#         X_train, y_train,
#         validation_data=(X_test, y_test),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[mc, es],
#         verbose=1
#     )

#     # Save scaler
#     with open(save_scaler_path, "wb") as f:
#         pickle.dump(scaler, f)

#     print("Saved LSTM model to:", save_model_path)
#     print("Saved scaler to:", save_scaler_path)

#     return save_model_path, save_scaler_path


# if __name__ == "__main__":
#     train_lstm(csv_path="data/RELIANCE_NS_stock_data.csv", epochs=30)


# scripts/train_lstm.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pickle
from src.preprocess import prepare_lstm_data
from src.lstm_model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_lstm(csv_path="C:/Users/GANESH V/stock-price-prediction/data/RELIANCE_NS_stock_data.csv",
               seq_len=120,
               epochs=40,
               batch_size=32,
               save_model_path="models/lstm_model.h5",
               save_scaler_path="models/lstm_scaler.pkl"):

    os.makedirs("models", exist_ok=True)

    # Load & clean data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # Prepare LSTM data
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
        df, feature="Close", seq_len=seq_len, train_ratio=0.8
    )

    model = build_lstm_model((X_train.shape[1], 1))

    mc = ModelCheckpoint(save_model_path, monitor="val_loss", save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[mc, es],
        verbose=1
    )

    with open(save_scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved LSTM model to:", save_model_path)
    print("Saved scaler to:", save_scaler_path)

if __name__ == "__main__":
    train_lstm(
        csv_path="C:/Users/GANESH V/stock-price-prediction/data/RELIANCE_NS_stock_data.csv",
        seq_len=120,
        epochs=40,
        batch_size=32
    )
