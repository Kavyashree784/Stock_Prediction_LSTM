import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Download stock data
df = yf.download("GOOGL", start="2009-01-01", end="2023-01-01")
data = df[['Close']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create sequences
x, y = [], []
for i in range(100, len(scaled_data)):
    x.append(scaled_data[i-100:i])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x, y, epochs=5, batch_size=32)

# Save model
model.save("keras_model.h5")
print(" Model saved as keras_model.h5")
