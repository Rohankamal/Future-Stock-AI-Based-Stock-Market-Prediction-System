# portfolio_optimizer.py
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import streamlit as st

def get_historical_prices_for_portfolio(tickers, start_date, end_date):
    """
    Fetches historical 'Close' prices for a list of tickers from Yahoo Finance.
    Handles cases where data for a ticker might not be available.
    """
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            # Fetch data and select 'Close' price
            # Using 'Close' directly as 'Adj Close' might not always be present or desired
            # for simpler portfolio calculations, and to avoid AttributeError.
            df = yf.download(ticker, start=start_date, end=end_date)['Close']
            if not df.empty:
                data[ticker] = df
            else:
                st.warning(f"No data found for {ticker} in the specified date range.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue
    
    if data.empty:
        return None
    
    # Drop rows with any NaN values (where some tickers might have missing data)
    data.dropna(inplace=True)
    
    if data.empty:
        st.warning("After dropping missing data, no complete historical data available for portfolio optimization.")
        return None

    return data

def optimize_portfolio(prices_df, risk_aversion=5):
    """
    Optimizes portfolio weights to maximize returns for a given risk aversion.
    Uses daily returns for calculation.
    """
    if prices_df.empty:
        return None

    returns = prices_df.pct_change().dropna()
    if returns.empty:
        st.warning("Not enough data to calculate returns for portfolio optimization.")
        return None

    num_assets = len(returns.columns)
    if num_assets < 2:
        st.warning("Portfolio optimization requires at least two assets with valid data.")
        return None

    # Calculate expected annual returns and covariance matrix
    # Annualize by multiplying by 252 (trading days in a year)
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_performance(weights, expected_returns, cov_matrix):
        """Calculates portfolio return, volatility, and (negative) Sharpe ratio."""
        weights = np.array(weights)
        p_return = np.sum(weights * expected_returns)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Assume a risk-free rate of 0 for simplicity, or define it globally
        risk_free_rate = 0.01 # Example risk-free rate (1%)
        sharpe_ratio = (p_return - risk_free_rate) / p_volatility
        
        # We want to maximize Sharpe ratio, so minimize its negative
        return np.array([p_return, p_volatility, sharpe_ratio])

    def objective_function(weights, expected_returns, cov_matrix, risk_aversion):
        """
        Objective function to minimize for portfolio optimization.
        Combines negative return (to maximize return) and volatility (to minimize risk).
        Risk aversion factor controls the balance.
        """
        metrics = portfolio_performance(weights, expected_returns, cov_matrix)
        p_return = metrics[0]
        p_volatility = metrics[1]
        
        # Objective: Maximize return and minimize volatility.
        # A higher risk_aversion means we penalize volatility more.
        # We minimize (negative return + risk_aversion * volatility)
        return -(p_return - risk_aversion * p_volatility) # Maximize return - risk_aversion * volatility

    # Constraints: sum of weights = 1, and each weight >= 0 (no short selling)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess: equally weighted portfolio
    initial_weights = num_assets * [1. / num_assets,]

    try:
        # Use minimize with the custom objective function
        optimized_results = sco.minimize(
            objective_function,
            initial_weights,
            args=(expected_returns, cov_matrix, risk_aversion),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if optimized_results.success:
            optimized_weights = pd.Series(optimized_results.x, index=returns.columns)
            return optimized_weights
        else:
            st.error(f"Optimization failed: {optimized_results.message}")
            return None
    except Exception as e:
        st.error(f"An error occurred during portfolio optimization: {e}")
        return None


def get_portfolio_performance(weights, prices_df):
    """
    Calculates the annualized expected return, volatility, and Sharpe ratio
    for a given portfolio and historical price data.
    """
    if prices_df.empty or weights is None:
        return None

    returns = prices_df.pct_change().dropna()
    if returns.empty:
        return None

    # Ensure weights are aligned with the columns of returns
    # This is crucial if some tickers were dropped during data fetching
    aligned_weights = weights[returns.columns]
    
    # Annualize by multiplying by 252 (trading days in a year)
    expected_portfolio_return = np.sum(aligned_weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(aligned_weights.T, np.dot(returns.cov() * 252, aligned_weights)))
    
    risk_free_rate = 0.01 # Example risk-free rate (1%)
    sharpe_ratio = (expected_portfolio_return - risk_free_rate) / portfolio_volatility

    return expected_portfolio_return, portfolio_volatility, sharpe_ratio

