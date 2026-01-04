# # src/preprocess.py
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# def compute_log_returns(df, price_col="Close"):
#     df = df.copy()
#     df["Log_Return"] = np.log(df[price_col] / df[price_col].shift(1))
#     df = df.dropna().reset_index(drop=True)
#     return df

# def prepare_lstm_data(df, feature, seq_len=60, train_ratio=0.8):
#     data = df[feature].values.reshape(-1, 1).astype(float)

#     train_stop = int(len(data) * train_ratio)

#     scaler = MinMaxScaler()
#     scaler.fit(data[:train_stop])

#     scaled = scaler.transform(data)

#     X, y = [], []
#     for i in range(seq_len, len(scaled)):
#         X.append(scaled[i-seq_len:i, 0])
#         y.append(scaled[i, 0])

#     X = np.array(X)
#     y = np.array(y).reshape(-1, 1)

#     cut = train_stop - seq_len
#     if cut <= 0:
#         cut = int(len(X) * train_ratio)

#     X_train, X_test = X[:cut], X[cut:]
#     y_train, y_test = y[:cut], y[cut:]

#     X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
#     X_test = X_test.reshape((X_test.shape[0], seq_len, 1))

#     return X_train, X_test, y_train, y_test, scaler


# src/preprocess.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, feature="Close", seq_len=60, train_ratio=0.8):
    # df must have a feature column (Close) as numeric and sorted by Date
    data = df[feature].values.reshape(-1,1).astype(float)
    n_total = len(data)
    train_stop = int(n_total * train_ratio)

    scaler = MinMaxScaler()
    scaler.fit(data[:train_stop])                 # fit only on train

    scaled = scaler.transform(data)

    X = []
    y = []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    # determine sequence split for train/test
    seq_train_cut = train_stop - seq_len
    if seq_train_cut <= 0:
        seq_train_cut = int(len(X) * train_ratio)

    X_train, X_test = X[:seq_train_cut], X[seq_train_cut:]
    y_train, y_test = y[:seq_train_cut], y[seq_train_cut:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler
