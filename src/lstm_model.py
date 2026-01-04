# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# def build_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mse")
#     return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape,
                     units=80,
                     dropout_rate=0.4):
    """
    Builds and returns an LSTM model for stock price prediction.

    Parameters:
    - input_shape: (sequence_length, features)
    - units: number of LSTM neurons
    - dropout_rate: dropout to prevent overfitting
    """

    model = Sequential()

    # First LSTM layer (returns sequences)
    model.add(
        LSTM(
            units,
            return_sequences=True,
            input_shape=input_shape
        )
    )
    model.add(Dropout(dropout_rate))

    # Second LSTM layer
    model.add(
        LSTM(units)
    )
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
