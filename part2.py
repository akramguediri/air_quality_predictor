import pandas as pd
import numpy as np
import tensorflow as tf

# Load the CSV data into a DataFrame
df = pd.read_csv('data.csv')

# Preprocess the timestamp column to make it suitable for the model
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Feature engineering (you can add more features as needed)
df['previous_main_aqi'] = df['main_aqi'].shift(1)
df['previous_co'] = df['co'].shift(1)
df['previous_no'] = df['no'].shift(1)
df['previous_no2'] = df['no2'].shift(1)
df['previous_o3'] = df['o3'].shift(1)
df['previous_so2'] = df['so2'].shift(1)
df['previous_pm2_5'] = df['pm2_5'].shift(1)
df['previous_pm10'] = df['pm10'].shift(1)

# Define your features and target
X = df[[
    'previous_main_aqi', 'previous_co', 'previous_no',
    'previous_no2', 'previous_o3', 'previous_so2',
    'previous_pm2_5', 'previous_pm10'
]].values

# Load the trained model
model = tf.keras.models.load_model('pollution_model.h5')

# Get the last known values as input for the first prediction
initial_features = X[-1].reshape(1, 8)  # Reshape to match the model input shape

# Create lists to store results
future_features = initial_features
predicted_results = []

# Predict for the next 10 days
for day in range(10):
    future_predictions = model.predict(future_features)  # Predict the next day's values

    # Store the current prediction
    predicted_results.append(future_predictions)

    # Prepare features for the next day's prediction
    future_features = future_predictions.reshape(1, 8)  # Reshape for the next prediction

# Create a DataFrame with the results for 10 days
results = pd.DataFrame(np.array(predicted_results).reshape(10, 8), columns=[
    'Main_AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10'
])

# Save the results to a CSV file
results.to_csv('predicted_results_10_days.csv', index=False)

