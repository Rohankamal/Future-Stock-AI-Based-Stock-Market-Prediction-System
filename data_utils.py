# data_utils.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st # Used for st.cache_data and st.error

@st.cache_data
def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetches historical stock data for a given ticker and date range.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame | None: DataFrame with historical data or None if data fetching fails.
    """
    try:
        # FIX: Added auto_adjust=True to silence the FutureWarning from yfinance
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            st.error(f"Sorry! No data found for '{ticker}'. Please enter a valid stock ticker.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}. Please check the ticker symbol or your internet connection.")
        return None

def prepare_data_for_lstm(data: pd.DataFrame, look_back: int = 60) -> tuple[np.ndarray, MinMaxScaler, np.ndarray]:
    """
    Prepares the closing price data for LSTM training and prediction.

    Args:
        data (pd.DataFrame): DataFrame containing stock data, must have a 'Close' column.
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple[np.ndarray, MinMaxScaler, np.ndarray]:
            - scaled_data (np.ndarray): The entire dataset scaled.
            - scaler (MinMaxScaler): The fitted scaler object.
            - last_n_days_scaled (np.ndarray): The last 'look_back' days of scaled data.
    """
    # Extract the 'Close' prices and reshape for scaling
    data_to_scale = data['Close'].values.reshape(-1, 1)

    # Initialize and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

    # Get the last 'look_back' days for initial prediction input
    last_n_days_scaled = scaled_data[len(scaled_data) - look_back:, :]

    return scaled_data, scaler, last_n_days_scaled

def create_dataset(data_scaled: np.ndarray, look_back: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates X (features) and y (target) datasets for LSTM training.

    Args:
        data_scaled (np.ndarray): Scaled historical data.
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and y datasets.
    """
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape for LSTM [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# IMPORTANT: If you have a get_historical_prices_for_portfolio function in your data_utils.py
# (which is imported by app.py for the Portfolio Suggestion tab),
# ensure it also has auto_adjust=True in its yf.download call.
# Example (add this if it's missing or update if it exists):
# def get_historical_prices_for_portfolio(tickers, start_date, end_date):
#     data = pd.DataFrame()
#     for ticker in tickers:
#         try:
#             df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True) # FIX HERE
#             if not df.empty:
#                 data[ticker] = df['Close']
#         except Exception as e:
#             print(f"Error fetching data for {ticker}: {e}")
#     data.dropna(inplace=True)
#     return data if not data.empty else None

