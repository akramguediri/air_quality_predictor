
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

y = df[[
    'main_aqi', 'co', 'no',
    'no2', 'o3', 'so2',
    'pm2_5', 'pm10'
]].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a feedforward neural network model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(64, activation='sigmoid'),  # Hidden layer
    tf.keras.layers.Dense(y_train.shape[1])  # Output layer for multiple variables
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Make predictions for future timestamps
future_features = np.array([
    df[[
        'previous_main_aqi', 'previous_co', 'previous_no',
        'previous_no2', 'previous_o3', 'previous_so2',
        'previous_pm2_5', 'previous_pm10'
    ]].iloc[-1]
])  # Use the last known values as input

future_predictions = model.predict(future_features)

# Create a DataFrame with the results
results = pd.DataFrame({
    'Timestamp': ['2023-12-01T00:00:00'],
    'Main_AQI': [abs(int(future_predictions[0][0]))],
    'CO': [abs(future_predictions[0][1])],
    'NO': [abs(future_predictions[0][2])],
    'NO2': [abs(future_predictions[0][3])],
    'O3': [abs(future_predictions[0][4])],
    'SO2': [abs(future_predictions[0][5])],
    'PM2.5': [abs(future_predictions[0][6])],
    'PM10': [abs(future_predictions[0][7])]
})

# Save the results to a CSV file
results.to_csv('predicted_results.csv', index=False)

client = OpenAI(api_key="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")  # Replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key

