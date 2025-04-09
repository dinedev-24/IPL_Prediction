# train_gru_rw_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Simulate realistic training data
np.random.seed(42)
X = []
y = []

for _ in range(1000):
    runs = np.random.randint(1, 15, size=20)
    wickets = np.random.binomial(1, 0.3, size=20)
    
    input_seq = np.column_stack((runs, wickets))
    X.append(input_seq)
    
    final_score = np.sum(runs)
    y.append(final_score)

X = np.array(X)                  # Shape: (1000, 20, 2)
y = np.array(y).reshape(-1, 1)   # Shape: (1000, 1)

# Fit scalers
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

X_scaled = np.array([scaler_input.fit_transform(x) for x in X])
y_scaled = scaler_output.fit_transform(y)

# Save scalers
os.makedirs("match_live_predictor", exist_ok=True)
joblib.dump(scaler_input, "match_live_predictor/scaler_input_rw.save")
joblib.dump(scaler_output, "match_live_predictor/scaler_output_rw.save")

# Build GRU model
model = Sequential([
    Input(shape=(20, 2)),
    GRU(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=1)

# Save model
model.save("match_live_predictor/gru_score_predictor_rw.keras")
print("GRU model retrained and saved successfully!")

