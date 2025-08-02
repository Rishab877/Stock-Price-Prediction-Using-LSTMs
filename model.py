import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def preprocess(df, lookback=60):
    data = df[['close']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_and_save_lstm(df, lookback, model_path="lstm_aapl.h5"):
    X, y, scaler = preprocess(df, lookback)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10, verbose=1)
    model.save(model_path)
    return model, scaler

def load_or_train_lstm(df, lookback, model_path="lstm_aapl.h5"):
    if os.path.exists(model_path):
        model = load_model(model_path)
        _, _, scaler = preprocess(df, lookback)
    else:
        model, scaler = train_and_save_lstm(df, lookback, model_path)
    return model, scaler

def predict_future(df, lookback, pred_days, model, scaler):
    data = df[['close']]
    last_days = data[-lookback:].values
    last_scaled = scaler.transform(last_days)
    inputs = last_scaled.reshape((1, lookback, 1))
    preds = []
    current_input = np.copy(inputs)
    for i in range(pred_days):
        pred_scaled = model.predict(current_input, verbose=0)
        pred = scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
        preds.append(pred)
        # update current_input for next prediction
        new_input = np.append(current_input[0, 1:, 0], pred_scaled[0][0])
        current_input = new_input.reshape((1, lookback, 1))
    return preds
