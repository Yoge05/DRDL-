# LSTM_Airline_Friend_Unique.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import urllib.request
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- DATA DOWNLOAD (NEW) ----------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
file = "airline_passengers.csv"

if not os.path.exists(file):
    urllib.request.urlretrieve(url, file)

data = pd.read_csv(file)
dataset = data.iloc[:,1].values.astype("float32")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(10,4))
plt.plot(dataset)
plt.title("International Airline Passengers")
plt.xlabel("Time (Months)")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

# ---------------- SCALING ----------------
dataset = dataset.reshape(-1,1)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

# ---------------- TRAIN-TEST SPLIT ----------------
train_size = int(len(dataset)*0.8)
train, test = dataset[:train_size], dataset[train_size:]

# ---------------- SEQUENCE GENERATION (CHANGED) ----------------
TIME_STEPS = 12

def create_sequences(data, steps):
    X, y = [], []
    for i in range(len(data)-steps):
        X.append(data[i:i+steps, 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

trainX, trainY = create_sequences(train, TIME_STEPS)
testX, testY = create_sequences(test, TIME_STEPS)

trainX = trainX.reshape(trainX.shape[0], TIME_STEPS, 1)
testX = testX.reshape(testX.shape[0], TIME_STEPS, 1)

# ---------------- MODEL (ARCHITECTURE CHANGED) ----------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEPS,1)),
    Dropout(0.2),
    LSTM(30),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ---------------- TRAINING ----------------
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(
    trainX, trainY,
    epochs=60,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- PREDICTION ----------------
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
trainY_inv = scaler.inverse_transform(trainY.reshape(-1,1))
testY_inv = scaler.inverse_transform(testY.reshape(-1,1))

# ---------------- METRICS ----------------
rmse = math.sqrt(mean_squared_error(testY_inv, test_pred))
mae = mean_absolute_error(testY_inv, test_pred)

print(f"\nTest RMSE: {rmse:.2f}")
print(f"Test MAE : {mae:.2f}")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(12,5))
plt.plot(scaler.inverse_transform(dataset), label="Actual Data")
plt.plot(range(TIME_STEPS, TIME_STEPS+len(train_pred)), train_pred, label="Train Prediction")
plt.plot(range(TIME_STEPS+len(train_pred), TIME_STEPS+len(train_pred)+len(test_pred)), test_pred, label="Test Prediction")
plt.legend()
plt.title("LSTM Airline Passenger Forecasting")
plt.show()

# ---------------- SAVE MODEL ----------------
model.save("friend_lstm_airline.h5")
print("Model saved as friend_lstm_airline.h5")
