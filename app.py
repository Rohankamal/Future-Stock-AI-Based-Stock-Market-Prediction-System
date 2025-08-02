import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import json
import time # For simulating real-time updates
import yfinance as yf # For fetching initial stock data

# Import db_utils for user authentication (removed portfolio DB functions)
from db_utils import init_db, add_user, verify_user

# --- Streamlit Page Configuration (MUST BE AT THE VERY TOP) ---
st.set_page_config(layout="wide", page_title="FutureStockAI")

# Initialize MySQL database for users (run once at app start)
init_db()

# --- Streamlit Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None # Will be set after login
if 'current_optimized_portfolio' not in st.session_state: # Only store current, not saved list
    st.session_state.current_optimized_portfolio = None
if 'live_chart_data' not in st.session_state:
    # Initialize live_chart_data with explicit dtypes to prevent future warnings
    st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object)
if 'current_live_ticker' not in st.session_state:
    st.session_state.current_live_ticker = None
# NEW: Store prediction data in session state
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None


# --- User Authentication Functions ---
def login_page():
    st.subheader("Login to FutureStockAI")
    email_input = st.text_input("Email", key="login_email_input")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        if verify_user(email_input, password): # Verify using email and password
            st.session_state.logged_in = True
            st.session_state.username = email_input # Display email as username
            st.session_state.user_id = email_input # Use email as user_id for MySQL
            st.success(f"Welcome, {email_input}!")
            st.rerun() # Re-added st.rerun() to force dashboard load
        else:
            st.error("Invalid email or password.")

def signup_page():
    st.subheader("Create a New Account")
    new_username = st.text_input("Choose a Username", key="signup_username")
    new_email = st.text_input("Email", key="signup_email")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

    if st.button("Sign Up", key="signup_button"):
        if new_username and new_email and new_password and confirm_password:
            if new_password == confirm_password:
                if add_user(new_username, new_email, new_password):
                    st.success("Account created successfully! You can now login.")
            else:
                st.error("Passwords do not match.")
        else:
            st.warning("Please fill in all required fields (Username, Email, Password).")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.current_optimized_portfolio = None # Clear current portfolio on logout
    st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object) # Clear live data on logout
    st.session_state.current_live_ticker = None
    st.session_state.prediction_df = None # Clear prediction data on logout
    st.success("Logged out successfully.")
    # st.rerun() # Re-added st.rerun() to force re-render to login page


# --- Main App Flow ---
if not st.session_state.logged_in:
    # This block now directly serves the authentication pages
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("Select an option:", ["Login", "Sign Up"], key="auth_option_radio")

    # Removed image and associated CSS/HTML for the landing page
    # st.image(image_url, caption='Welcome to FutureStockAI', use_container_width=True)
    st.markdown("---") # Keep the separator if desired

    if auth_option == "Login":
        login_page()
    else:
        signup_page()
else: # This block runs only if logged_in is True
    # Import functions from our custom modules
    from data_utils import get_stock_data, prepare_data_for_lstm, create_dataset
    from model_trainer import build_and_train_model, predict_future_prices
    from sentiment_analyzer import analyze_sentiment, get_sentiment_label, fetch_news_headlines, NEWSAPI_KEY
    from portfolio_optimizer import get_historical_prices_for_portfolio, optimize_portfolio, get_portfolio_performance
    # Removed import for chatbot_ai


    # --- Streamlit UI Configuration (Main App Content) ---
    st.sidebar.info(f"**Logged in as:** `{st.session_state.username}`")
    st.sidebar.button("Logout", on_click=logout, key="logout_button_sidebar")
    st.sidebar.markdown("---")


    # --- Tabs for Navigation (Only available when logged in) ---
    # Removed tab4 (AI Chatbot)
    tab1, tab2, tab3, tab4 = st.tabs(["Stock Prediction", "Sentiment Analysis", "Portfolio Suggestion", "Live Stock Tracking"])

    # --- Tab 1: Stock Price Prediction ---
    with tab1:
        st.header("📊 Stock Price Prediction")
        st.markdown("Enter a stock ticker to view historical data and predict future prices using an LSTM model.")

        # Sidebar for Stock Prediction Inputs
        st.sidebar.subheader("Stock Prediction Parameters")
        ticker_symbol_pred = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL", key="sidebar_ticker_pred").upper()
        today = datetime.date.today()
        end_date = today
        # FIX: Reduced historical data for faster training/prediction
        start_date = today - datetime.timedelta(days=365 * 2) # Changed from 5 years to 2 years of data
        num_prediction_days = st.sidebar.slider("Number of days to predict?", 1, 30, 7)

        # Model parameters (can be exposed to UI later for advanced users)
        look_back_period = 60 # Using last 60 days to predict the next day
        # FIX: Reduced training epochs for faster training
        training_epochs = 20 # Changed from 50 to 20 epochs
        training_batch_size = 32

        if st.button("Run Stock Prediction", key="run_stock_pred"): # Unique key for button
            if not ticker_symbol_pred:
                st.error("Please enter a stock ticker symbol for prediction.")
            else:
                st.subheader(f"Data and Prediction for {ticker_symbol_pred}")

                # 1. Get Data
                with st.spinner(f"Fetching historical data for '{ticker_symbol_pred}'..."):
                    # FIX: Add auto_adjust=True to yf.download calls
                    stock_data = get_stock_data(ticker_symbol_pred, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                if stock_data is not None and not stock_data.empty: # Added check for empty stock_data
                    # Display Latest Closing Price
                    # FIX: Use .item() to extract scalar from Series
                    latest_close_price = stock_data['Close'].iloc[-1].item()
                    latest_close_date = stock_data.index[-1].strftime('%Y-%m-%d')
                    st.info(f"**Latest Closing Price ({latest_close_date}):** ${latest_close_price:.2f}")
                    st.markdown("*(Prediction model uses data up to this point.)*")

                    # 2. Prepare Data for LSTM
                    with st.spinner("Preparing data..."):
                        scaled_data, scaler, last_60_days_for_prediction = prepare_data_for_lstm(stock_data, look_back=look_back_period)

                        training_data_length = int(len(scaled_data) * 0.8)
                        train_data_scaled = scaled_data[0:training_data_length, :]
                        X_train, y_train = create_dataset(train_data_scaled, look_back=look_back_period)

                    # 3. Build and Train Model (Always standard LSTM now)
                    # FIX: The UserWarning about input_shape will be handled in model_trainer.py
                    model = build_and_train_model(X_train, y_train, epochs=training_epochs, batch_size=training_batch_size)

                    # 4. Make Predictions
                    with st.spinner(f"Generating predictions for the next {num_prediction_days} days..."):
                        predicted_future_prices = predict_future_prices(model, last_60_days_for_prediction, scaler, num_prediction_days)

                    # Create date range for predictions
                    last_actual_date = stock_data.index[-1]
                    prediction_dates = [last_actual_date + datetime.timedelta(days=i) for i in range(1, num_prediction_days + 1)]

                    # Store prediction_df in session state
                    st.session_state.prediction_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted Close': predicted_future_prices.flatten() # Ensure 1D numpy array
                    })
                    st.session_state.prediction_df.set_index('Date', inplace=True)
                    # Store stock_data for plotting after rerun
                    st.session_state.last_stock_data_for_pred = stock_data
                else:
                    st.error(f"Could not retrieve historical data for '{ticker_symbol_pred}'. Please check the ticker symbol or try a different date range.")
                    st.session_state.prediction_df = None # Clear previous prediction if new fetch fails
                    st.session_state.last_stock_data_for_pred = pd.DataFrame() # Clear stock data for plot


        # --- Display Stock Prediction Results (moved outside the button block) ---
        if st.session_state.prediction_df is not None and not st.session_state.prediction_df.empty:
            st.write("---")
            # Use ticker_symbol_pred from input, as it's consistent across reruns for this tab
            st.subheader(f"Price Prediction for {ticker_symbol_pred} (Next {num_prediction_days} Days)") 

            # To plot, we need recent actual prices and dates.
            # Use data stored in session state
            recent_actual_prices = st.session_state.last_stock_data_for_pred['Close'].tail(look_back_period).values if not st.session_state.last_stock_data_for_pred.empty else np.array([])
            recent_actual_dates = st.session_state.last_stock_data_for_pred.index[-look_back_period:] if not st.session_state.last_stock_data_for_pred.empty else pd.DatetimeIndex([])

            if len(recent_actual_prices) > 0 and len(st.session_state.prediction_df) > 0:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=recent_actual_dates, y=recent_actual_prices.flatten(), # Ensure 1D
                                             mode='lines', name='Actual Prices',
                                             line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=st.session_state.prediction_df.index, y=st.session_state.prediction_df['Predicted Close'].values.flatten(), # Ensure 1D
                                             mode='lines', name='Predicted Prices',
                                             line=dict(color='red', dash='dot', width=2)))

                # Dynamically set y-axis range for prediction chart
                all_prices = np.concatenate((recent_actual_prices.flatten(), st.session_state.prediction_df['Predicted Close'].values.flatten()))
                min_all_price = np.min(all_prices)
                max_all_price = np.max(all_prices)

                fig_pred.update_layout(
                    title_text=f'{ticker_symbol_pred} Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500,
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis_range=[min_all_price * 0.95, max_all_price * 1.05] # Add 5% buffer
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning("Not enough data to display prediction chart. Please click 'Run Stock Prediction'.")


            st.write("---")
            st.subheader("Predicted Prices Table")
            st.dataframe(st.session_state.prediction_df.style.format({"Predicted Close": "{:.2f}"}), use_container_width=True)

            # CSV Export for Predictions
            csv_data = st.session_state.prediction_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Prediction Data as CSV",
                data=csv_data,
                file_name=f"{ticker_symbol_pred}_predictions.csv",
                mime="text/csv",
                key="download_predictions_csv"
            )
        else:
            st.info("Click 'Run Stock Prediction' to generate and view predictions.")


    # --- Tab 2: Sentiment Analysis ---
    with tab2:
        st.header("💬 Sentiment Analysis")
        st.markdown("Get real-time market sentiment by analyzing recent news headlines for a stock.")

        if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
            st.warning("⚠️ **NewsAPI Key Missing!** Please get your free API key from [NewsAPI.org](https://newsapi.org/) and replace 'YOUR_NEWSAPI_KEY' in `sentiment_analyzer.py` to enable this feature.")
        else:
            sentiment_ticker_input = st.text_input("Enter Stock Ticker for News Sentiment (e.g., TSLA, MSFT):", "TSLA", key="sentiment_ticker_input").upper()
            num_news_days_back = st.slider("Look back for news (days):", 1, 30, 7, key="num_news_days_back")

            if st.button("Fetch & Analyze News Sentiment", key="fetch_analyze_sentiment_btn"):
                if sentiment_ticker_input:
                    with st.spinner(f"Fetching news for '{sentiment_ticker_input}' and analyzing sentiment..."):
                        articles = fetch_news_headlines(sentiment_ticker_input, NEWSAPI_KEY, num_news_days_back)

                        if articles:
                            st.subheader(f"Recent News Headlines for {sentiment_ticker_input}")
                            all_compound_scores = []

                            for i, article in enumerate(articles[:10]): # Limit to first 10 articles for display
                                title = article.get('title', 'No Title')
                                description = article.get('description', 'No Description')
                                url = article.get('url', '#')
                                source = article.get('source', {}).get('name', 'Unknown Source')
                                published_at = article.get('publishedAt', 'N/A')

                                sentiment_scores = analyze_sentiment(title + " " + (description if description else ""))
                                compound_score = sentiment_scores['compound']
                                sentiment_label = get_sentiment_label(compound_score)
                                all_compound_scores.append(compound_score)

                                st.markdown(f"**[{title}]({url})**")
                                st.write(f"Source: {source} | Published: {published_at}")
                                st.write(f"Sentiment: **{sentiment_label}** (Compound: {compound_score:.2f})")
                                st.markdown(f"*{description}*")
                                st.markdown("---")

                            if all_compound_scores:
                                average_compound_score = np.mean(all_compound_scores)
                                overall_sentiment_label = get_sentiment_label(average_compound_score)
                                st.subheader("Overall Sentiment Summary")
                                st.write(f"**Average Compound Score:** {average_compound_score:.2f}")
                                st.write(f"**Overall Market Sentiment:** {overall_sentiment_label}")

                                if overall_sentiment_label == "Positive":
                                    st.success("😊 The overall news sentiment for this stock is positive.")
                                elif overall_sentiment_label == "Negative":
                                    st.error("😠 The overall news sentiment for this stock is negative.")
                                else:
                                    st.info("😐 The overall news sentiment for this stock is neutral.")
                            else:
                                st.info(f"No headlines found for '{sentiment_ticker_input}' in the last {num_news_days_back} days or unable to analyze.")
                        else:
                            st.info(f"No news headlines found for '{sentiment_ticker_input}' in the last {num_news_days_back} days.")
                else:
                    st.warning("Please enter a stock ticker for news sentiment analysis.")

    # --- Tab 3: Portfolio Suggestion ---
    with tab3:
        st.header("💰 Portfolio Suggestion (Optimized)")
        st.markdown("Enter a list of stocks you are interested in, and we will suggest optimal weights to maximize returns for a given risk level.")

        portfolio_tickers_input = st.text_input(
            "Enter stock tickers (comma-separated, e.g., AAPL, MSFT, GOOGL, AMZN):",
            "AAPL, MSFT, GOOGL",
            key="portfolio_tickers_input"
        )
        portfolio_tickers_list = [ticker.strip().upper() for ticker in portfolio_tickers_input.split(',') if ticker.strip()]

        risk_tolerance_level = st.slider(
            "Your Risk Tolerance (1 = Very Low Risk, 10 = Very High Risk):",
            1, 10, 5, key="risk_tolerance_level_slider"
        )

        col_optimize, col_save = st.columns([0.7, 0.3])

        with col_optimize:
            if st.button("Optimize Portfolio", key="optimize_portfolio_btn"):
                if not portfolio_tickers_list or len(portfolio_tickers_list) < 2:
                    st.warning("Please enter at least two stock tickers to optimize a portfolio.")
                else:
                    portfolio_end_date = datetime.date.today()
                    portfolio_start_date = portfolio_end_date - datetime.timedelta(days=365 * 3) # 3 years for portfolio analysis

                    with st.spinner("Fetching historical data and optimizing portfolio..."):
                        # FIX: get_historical_prices_for_portfolio in data_utils.py should also use auto_adjust=True
                        prices_df = get_historical_prices_for_portfolio(
                            portfolio_tickers_list,
                            portfolio_start_date.strftime('%Y-%m-%d'),
                            portfolio_end_date.strftime('%Y-%m-%d')
                        )

                        if prices_df is not None and not prices_df.empty:
                            available_tickers = prices_df.columns.tolist()
                            if len(available_tickers) < len(portfolio_tickers_list):
                                st.warning(f"Could not fetch data for some tickers. Optimizing with: {', '.join(available_tickers)}")
                            if len(available_tickers) < 2:
                                st.error("Not enough valid tickers with historical data to perform optimization. Please check the tickers.")
                            else:
                                optimized_weights = optimize_portfolio(prices_df, risk_aversion=risk_tolerance_level)

                                if optimized_weights is not None and not optimized_weights.empty:
                                    st.session_state.current_optimized_portfolio = { # Store current optimized portfolio
                                        'weights': optimized_weights, # This is likely a Pandas Series
                                        'performance': get_portfolio_performance(optimized_weights, prices_df),
                                        'tickers': portfolio_tickers_list,
                                        'risk_tolerance': risk_tolerance_level
                                    }
                                    st.success("Portfolio optimized!")
                                    # Removed st.rerun()
                                else:
                                    st.warning("Portfolio optimization failed. Check your ticker symbols or try different ones.")
                        else:
                            st.error("Could not retrieve sufficient historical data for portfolio optimization. Please check the tickers.")

        # Display Current Optimized Portfolio (if available in session state)
        if 'current_optimized_portfolio' in st.session_state and st.session_state.current_optimized_portfolio:
            current_weights = st.session_state.current_optimized_portfolio['weights']
            current_performance = st.session_state.current_optimized_portfolio['performance']
            current_tickers = st.session_state.current_optimized_portfolio['tickers']
            current_risk_tolerance = st.session_state.current_optimized_portfolio['risk_tolerance']
            # Removed current_portfolio_name_display, as it was only used for download which is removed.

            st.write("---")
            st.subheader(f"Current Optimized Portfolio for {current_tickers} (Risk: {current_risk_tolerance}/10)")

            # FIX: Ensure current_weights is a dictionary for DataFrame creation
            if current_weights is not None:
                if isinstance(current_weights, pd.Series):
                    weights_dict_for_display = current_weights.to_dict()
                elif isinstance(current_weights, dict):
                    weights_dict_for_display = current_weights
                else:
                    st.warning("Optimized weights are in an unexpected format. Cannot display.")
                    weights_dict_for_display = {} # Fallback to empty dict

                if weights_dict_for_display: # Check if the dictionary is not empty
                    weights_df = pd.DataFrame(list(weights_dict_for_display.items()), columns=['Ticker', 'Weight'])
                    weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
                    st.dataframe(weights_df[['Ticker', 'Weight (%)']], use_container_width=True)
                else:
                    st.info("No optimized portfolio weights to display.")

                if current_performance:
                    expected_return, annual_volatility, sharpe_ratio = current_performance
                    st.subheader("Performance Metrics (Annualized)")
                    st.markdown(f"**Expected Annual Return:** {expected_return * 100:.2f}%")
                    st.markdown(f"**Annual Volatility (Risk):** {annual_volatility * 100:.2f}%")
                    st.markdown(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                    st.info("Higher Sharpe Ratio indicates better risk-adjusted returns.")
                else:
                    st.warning("Could not calculate portfolio performance for the current optimized portfolio.")

                # Removed the "Download Current Portfolio" section completely
            else: # This else corresponds to `if current_weights is not None:`
                st.info("Optimize a portfolio first to see its details.") # Changed message
        else: # This else corresponds to `if 'current_optimized_portfolio' in st.session_state ...`
            st.info("Optimize a portfolio to see its details.") # Changed message

    # --- Tab 4: Live Stock Tracking ---
    with tab4:
        st.header("📈 Live Stock Tracking")
        st.markdown("Track real-time (simulated) stock prices and visualize movements for any stock ticker.")


        col_select, col_refresh_rate = st.columns([0.7, 0.3])

        with col_select:
            # Dropdown for popular tickers
            default_tickers = [
                "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "GOOG", "NFLX", # US Tech Giants
                "BRK-B", "JPM", "V", "PG", "JNJ", "UNH", "XOM", "CVX", # US Blue Chips
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", # Indian Stocks
                "TATASTEEL.NS", "LT.NS", "MARUTI.NS", "ASIANPAINT.NS", "ADANIENT.NS", # More Indian Stocks
                "HSBC", "SAP", "TM", "SNEJF", "BABA", "TCEHY", "NSRGY", "RY.TO", # International
                "SPY", "QQQ", "DIA", "GLD", "SLV" # ETFs
            ]
            default_tickers.sort() # Sort alphabetically for better UX

            # Removed custom ticker input
            # custom_ticker = st.text_input("Or enter a custom ticker:", "AAPL", key="custom_live_ticker").upper()

            # The selected_ticker will always come from the selectbox
            # No need for is_selectbox_disabled or final_ticker logic as there's no custom input
            selected_ticker = st.selectbox(
                "Select a stock ticker:",
                options=default_tickers,
                index=default_tickers.index(st.session_state.current_live_ticker) if st.session_state.current_live_ticker in default_tickers else 0,
                key="live_ticker_select"
            )
            
            final_ticker = selected_ticker # Now final_ticker is simply the selected_ticker

        with col_refresh_rate:
            refresh_interval = st.slider("Refresh Interval (seconds):", 1, 10, 5, key="refresh_interval_slider")

        if st.button("Start Live Tracking", key="start_live_tracking_btn"):
            if not final_ticker:
                st.warning("Please select a stock ticker to start tracking.")
            else:
                st.session_state.current_live_ticker = final_ticker
                # Initialize live_chart_data with explicit dtypes to prevent future warnings
                st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object)

                st.subheader(f"Live Price for {final_ticker}")
                price_metric_placeholder = st.empty()
                chart_placeholder = st.empty()

                # Fetch initial historical data for better simulation start
                try:
                    # Fetch data for the last 2 days at 1-minute interval
                    # FIX: Add auto_adjust=True
                    initial_data = yf.download(final_ticker, period="2d", interval="1m", auto_adjust=True)
                    if not initial_data.empty:
                        # FIX: Use .item() to extract scalar from Series
                        last_price = initial_data['Close'].iloc[-1].item()
                        
                        # FIX: Ensure initial_data index is timezone-naive for consistency
                        if initial_data.index.tz is not None:
                            initial_data.index = initial_data.index.tz_localize(None)

                        # Initialize live_chart_data with recent historical minute data
                        # FIX: Ensure dtypes are consistent for concatenation
                        new_initial_data = pd.DataFrame({
                            'Time': initial_data.index,
                            'Price': initial_data['Close'].values.flatten()
                        })
                        st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_initial_data], ignore_index=True)

                        # Keep only the last 100 points for initial display
                        if len(st.session_state.live_chart_data) > 100:
                            st.session_state.live_chart_data = st.session_state.live_chart_data.tail(100).reset_index(drop=True)
                    else:
                        last_price = 100.0 # Default if no data
                        st.warning(f"Could not retrieve initial data for '{final_ticker}'. Starting with a default price.")
                except Exception as e:
                    last_price = 100.0 # Default if any error
                    st.error(f"Error retrieving initial data for '{final_ticker}': {e}. Starting with a default price.")
                    st.warning("Please note: Live data is simulated. For real-time data, a dedicated API key and backend are needed.")

                # Loop for continuous updates
                while True:
                    current_time = datetime.datetime.now() # This is timezone-naive

                    new_price = last_price # Start with the last known price

                    # Attempt to fetch the very latest minute data from yfinance
                    try:
                        # Changed period from "1m" to "1d" to be valid for 1m interval
                        # FIX: Add auto_adjust=True
                        latest_data = yf.download(final_ticker, period="1d", interval="1m", auto_adjust=True)
                        if not latest_data.empty:
                            # FIX: Ensure latest_data index is timezone-naive for comparison
                            if latest_data.index.tz is not None:
                                latest_data.index = latest_data.index.tz_localize(None)

                            # If new data is available, use it
                            if not st.session_state.live_chart_data.empty and latest_data.index[-1] > st.session_state.live_chart_data['Time'].max():
                                new_price = latest_data['Close'].iloc[-1].item() # FIX: Use .item()
                                # Append the new data point
                                new_data_point = pd.DataFrame([{'Time': latest_data.index[-1], 'Price': new_price}])
                                # FIX: Ensure dtypes are consistent for concatenation
                                st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)
                            else:
                                # If no new minute data, simulate a small fluctuation
                                price_change = np.random.uniform(-0.5, 0.5)
                                new_price = last_price + price_change # FIX: Removed redundant float() call, as last_price is already float
                                if new_price < 0: new_price = 0.01
                                new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                                # FIX: Ensure dtypes are consistent for concatenation
                                st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)
                        else:
                            # If yfinance returns empty for 1m interval, simulate
                            price_change = np.random.uniform(-0.5, 0.5)
                            new_price = last_price + price_change # FIX: Removed redundant float() call
                            if new_price < 0: new_price = 0.01
                            new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                            # FIX: Ensure dtypes are consistent for concatenation
                            st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)

                    except Exception as e:
                        # Fallback to simulation if fetching fails
                        st.warning(f"Error retrieving latest data for '{final_ticker}': {e}. Simulating price.")
                        price_change = np.random.uniform(-0.5, 0.5)
                        new_price = last_price + price_change # FIX: Removed redundant float() call
                        if new_price < 0: new_price = 0.01
                        new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                        # FIX: Ensure dtypes are consistent for concatenation
                        st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)


                    # Update metric
                    price_metric_placeholder.metric(label=f"Current Price ({final_ticker})", value=f"${new_price:.2f}", delta=f"{new_price - last_price:.2f}")

                    # Keep only the last 100 points for performance
                    if len(st.session_state.live_chart_data) > 100:
                        st.session_state.live_chart_data = st.session_state.live_chart_data.tail(100).reset_index(drop=True)

                    # Update chart
                    chart_placeholder.line_chart(st.session_state.live_chart_data.set_index('Time'))

                    last_price = new_price
                    time.sleep(refresh_interval)
