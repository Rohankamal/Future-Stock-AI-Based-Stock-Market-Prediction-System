# model_trainer.py
import numpy as np
from tensorflow.keras.models import Sequential
# FIX: Import Input layer from tensorflow.keras.layers
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import autokeras as ak # NEW IMPORT for AutoML

@st.cache_resource
def build_and_train_model(X_train: np.ndarray, y_train: np.ndarray,
                          epochs: int = 50, batch_size: int = 32) -> Sequential:
    """
    Builds and trains a standard LSTM model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Sequential: The trained Keras LSTM model.
    """
    st.write("Training standard LSTM model... This might take a moment.")

    model = Sequential()
    # FIX: Use Input layer explicitly as the first layer
    # input_shape should be (timesteps, features)
    model.add(Input(shape=(X_train.shape[1], 1)))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    st.success("Standard LSTM model training complete!")
    return model

@st.cache_resource
def build_and_train_automl_model(X_train: np.ndarray, y_train: np.ndarray,
                                 max_trials: int = 5) -> Sequential:
    """
    Builds and trains an AutoML model using AutoKeras.

    Args:
        X_train (np.ndarray): Training features (reshaped to 2D for StructuredDataRegressor).
        y_train (np.ndarray): Training targets.
        max_trials (int): The number of different Keras Models to try.

    Returns:
        Sequential: The best model found by AutoKeras.
    """
    st.write(f"Training AutoML model with {max_trials} trials... This will take longer.")
    st.warning("AutoML is computationally intensive and may take several minutes depending on data size and trials.")

    # AutoKeras StructuredDataRegressor expects 2D input for X
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1) # Flatten the look_back_period dimension

    # Initialize the StructuredDataRegressor
    # project_name and directory are important for caching and resuming trials
    reg = ak.StructuredDataRegressor(
        overwrite=True, # Overwrite previous trials
        max_trials=max_trials, # Number of different models to try
        metrics=['mse'], # Use Mean Squared Error as metric
        objective='val_mse', # Objective to minimize during search
        directory='autokeras_cache', # Directory to store temporary files and models
        seed=42 # For reproducibility
    )

    # Search for the best model
    reg.fit(X_train_reshaped, y_train, verbose=0) # verbose=0 to hide training logs

    # Export the best model found
    best_model = reg.export_model()
    best_model.compile(optimizer='adam', loss='mean_squared_error') # Recompile the exported model
    st.success("AutoML model training complete!")
    return best_model


def predict_future_prices(model: Sequential, last_n_days_scaled: np.ndarray,
                          scaler: MinMaxScaler, num_prediction_days: int = 7,
                          is_automl_model: bool = False) -> np.ndarray:
    """
    Predicts future stock prices using the trained model (LSTM or AutoML).

    Args:
        model (Sequential): The trained Keras (LSTM or AutoML) model.
        last_n_days_scaled (np.ndarray): The last 'look_back' days of scaled data.
        scaler (MinMaxScaler): The scaler used for inverse transformation.
        num_prediction_days (int): Number of days to predict into the future.
        is_automl_model (bool): True if the model is an AutoKeras model, False for standard LSTM.

    Returns:
        np.ndarray: Flattened array of predicted prices.
    """
    # Reshape the input for the model: [1, look_back, 1] for LSTM
    # or [1, look_back] for AutoML StructuredDataRegressor
    current_input = last_n_days_scaled

    predicted_prices_scaled = []

    for _ in range(num_prediction_days):
        if is_automl_model:
            # AutoKeras StructuredDataRegressor expects 2D input for prediction
            input_for_predict = current_input.reshape(1, -1)
        else:
            # Standard LSTM expects 3D input
            # Ensure current_input is 2D before reshaping to 3D for LSTM
            if current_input.ndim == 1: # If it's a 1D array from append
                current_input = current_input.reshape(-1, 1) # Make it (look_back, 1)
            input_for_predict = current_input.reshape(1, current_input.shape[0], 1)

        # Predict the next day's price
        next_day_prediction_scaled = model.predict(input_for_predict, verbose=0)[0, 0]
        predicted_prices_scaled.append(next_day_prediction_scaled)

        # Update the input sequence: remove the oldest prediction, add the new one
        # Ensure the appended value is treated as a single feature for the next iteration
        current_input = np.append(current_input[1:], next_day_prediction_scaled) # Append directly to 1D array

    # Inverse transform the scaled predictions to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    return predicted_prices.flatten()

