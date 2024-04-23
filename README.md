# Pollution Prediction Model

## Overview
This repository contains Python scripts for training a neural network model to predict pollution levels based on historical data, as well as making predictions for future pollution levels using the trained model.

Pollution prediction is crucial for environmental monitoring and public health management. By accurately forecasting pollution levels, authorities can take timely measures to mitigate the impact on air quality and human health.

This project was initiated during the Smart City Hackathon in Heilbronn in November 2023, with the aim of developing innovative solutions for urban sustainability and environmental monitoring.

## Files
- **data.csv**: CSV file containing historical pollution data. It includes columns for timestamps and various pollution-related features such as main_aqi, co, no, no2, o3, so2, pm2_5, and pm10.
- **pollution_model.py**: Python script for training the pollution prediction model. It loads the data, preprocesses it, engineers features, creates and trains a neural network model using TensorFlow, and saves the trained model to a file named 'pollution_model.h5'.
- **predict_pollution.py**: Python script for making predictions for future pollution levels using the trained model. It loads the trained model, prepares input features for prediction, iteratively predicts pollution levels for the next 10 days, and saves the predicted results to a CSV file named 'predicted_results_10_days.csv'.

## Usage
1. **Training the Model**: Run the 'pollution_model.py' script to train the pollution prediction model. Make sure to have the 'data.csv' file in the same directory. After running the script, a trained model will be saved as 'pollution_model.h5'.

2. **Making Predictions**: Once the model is trained, run the 'predict_pollution.py' script to make predictions for future pollution levels. Ensure that the trained model file 'pollution_model.h5' and the 'data.csv' file are present in the directory. After running the script, predicted pollution levels for the next 10 days will be saved in a CSV file named 'predicted_results_10_days.csv'.

## Dependencies
- pandas: Data manipulation and preprocessing.
- numpy: Numerical operations and array handling.
- TensorFlow: Deep learning framework for building and training neural network models.
- scikit-learn: Machine learning library for data splitting and model evaluation.

## Future Improvements
- Implement more advanced neural network architectures such as recurrent neural networks (RNNs) or long short-term memory (LSTM) networks for better capturing temporal dependencies in the data.
- Explore additional features or external factors (e.g., weather data, geographical information) that may influence pollution levels and incorporate them into the model for improved accuracy.
- Fine-tune hyperparameters and optimize the model architecture to achieve better performance and generalization on unseen data.

## Note
This is only one part of the project, the other parts (ui, Database, etc) are made by the other team members in the hackathon.
