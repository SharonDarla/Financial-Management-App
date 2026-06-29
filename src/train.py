import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Layer, Input, Multiply
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from attention import Attention

import warnings
from pmdarima import auto_arima # type: ignore


warnings.filterwarnings("ignore")

# -------------------------------
# Load & preprocess data
# -------------------------------
data = pd.read_csv("data20.csv")
data['Date'] = pd.to_datetime(data['Date'])
companies = data['Company'].unique()

# -------------------------------
# Scaling per company
# -------------------------------
scaler_dict = {}
for company in companies:
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = data[data['Company'] == company]['Close'].values.reshape(-1, 1)
    data.loc[data['Company'] == company, 'Scaled_Close'] = scaler.fit_transform(series)
    scaler_dict[company] = scaler

# -------------------------------
# Prepare LSTM/GRU dataset
# -------------------------------
def create_dataset(series, time_step=60):
    X, y = [], []
    for i in range(len(series) - time_step):
        X.append(series[i:i + time_step])
        y.append(series[i + time_step])
    return np.array(X), np.array(y)

time_step = 60
X_dict, y_dict = {}, {}

for company in companies:
    series = data[data['Company'] == company]['Scaled_Close'].values
    X, y = create_dataset(series, time_step)
    X_dict[company], y_dict[company] = X.reshape(-1, time_step, 1), y

# -------------------------------
# Model Builders
# -------------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(80, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(80),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------------
# Train models per company
# -------------------------------
pred_lstm_dict, pred_gru_dict, pred_arima_dict = {}, {}, {}
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

for company in companies:
    print(f"\n🔹 Training models for {company}...")
    X, y = X_dict[company], y_dict[company]
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- LSTM ---
    lstm_model = build_lstm((time_step, 1))
    lstm_model.fit(
        X_train, y_train,
        epochs=50, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    pred_lstm = lstm_model.predict(X_test)
    pred_lstm = scaler_dict[company].inverse_transform(pred_lstm)
    y_true_lstm = scaler_dict[company].inverse_transform(y_test.reshape(-1, 1))
    pred_lstm_dict[company] = (y_true_lstm.flatten(), pred_lstm.flatten())

    # --- GRU ---
    gru_model = build_gru((time_step, 1))
    gru_model.fit(
        X_train, y_train,
        epochs=60, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    pred_gru = gru_model.predict(X_test)
    pred_gru = scaler_dict[company].inverse_transform(pred_gru)
    y_true_gru = scaler_dict[company].inverse_transform(y_test.reshape(-1, 1))
    pred_gru_dict[company] = (y_true_gru.flatten(), pred_gru.flatten())

   

# -------------------------------
# Build Attention LSTM model
# -------------------------------
def build_attention_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = Attention()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------------
# Train Attention LSTM per company
# -------------------------------
pred_att_lstm_dict = {}

for company in companies:
    print(f"\n🔹 Training Attention LSTM for {company}...")
    X, y = X_dict[company], y_dict[company]
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_attention_lstm((time_step, 1))
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    pred = model.predict(X_test)
    pred = scaler_dict[company].inverse_transform(pred)
    y_true = scaler_dict[company].inverse_transform(y_test.reshape(-1, 1))
    pred_att_lstm_dict[company] = (y_true.flatten(), pred.flatten())

# -------------------------------
# Train ARIMA per company
# -------------------------------
for company in companies:
    print(f"\n🔹 Training ARIMA for {company}...")
    series = data[data['Company'] == company]['Scaled_Close'].values
    split = int(len(series) * 0.8)
    train_series = series[:split]
    test_series = series[split:]

    # Use auto_arima to find best ARIMA parameters
    arima_model = auto_arima(train_series, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
    arima_model.fit(train_series)

    # Forecast for the test period
    forecast = arima_model.predict(n_periods=len(test_series))
    pred_arima = scaler_dict[company].inverse_transform(forecast.reshape(-1, 1))
    y_true_arima = scaler_dict[company].inverse_transform(test_series.reshape(-1, 1))
    pred_arima_dict[company] = (y_true_arima.flatten(), pred_arima.flatten())
def evaluate_all(pred_dict, model_name):
    mse_list, rmse_list, mae_list, r2_list = [], [], [], []
    for company in companies:
        y_true, y_pred = pred_dict[company]
        mse_list.append(mean_squared_error(y_true, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_list.append(mean_absolute_error(y_true, y_pred))
        r2_list.append(r2_score(y_true, y_pred))

    print(f"\n--- {model_name} ---")
    print(f"MSE: {np.mean(mse_list):.4f}, RMSE: {np.mean(rmse_list):.4f}, "
          f"MAE: {np.mean(mae_list):.4f}, R2: {np.mean(r2_list):.4f}\n")

# -------------------------------
# Evaluate Models
# -------------------------------
evaluate_all(pred_lstm_dict, "LSTM")
evaluate_all(pred_gru_dict, "GRU (Fine-tuned)")
evaluate_all(pred_att_lstm_dict, "Attention LSTM")

# -------------------------------
# Save predictions for analytics/app
# -------------------------------
np.save("pred_lstm.npy", pred_lstm_dict)
np.save("pred_gru.npy", pred_gru_dict)
np.save("pred_att_lstm.npy", pred_att_lstm_dict)
# -------------------------------
