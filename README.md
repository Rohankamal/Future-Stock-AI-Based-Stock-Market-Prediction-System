FutureStockAI: AI Based Stock Market Prediction System
This project is a comprehensive AI-driven web application built with Streamlit, designed to empower users with tools for real-time market analysis, stock price forecasting, and data-backed investment decisions. The application demonstrates proficiency in modern data science libraries and full-stack development principles for a robust user experience.

🌟 Key Features & Technical Highlights
Optimized AI-Driven Prediction Engine: Utilizes a Long Short-Term Memory (LSTM) deep learning model for time-series forecasting of stock prices.

Performance: Optimized the training workflow by reducing the historical data window to 2 years and the number of training epochs to 20, achieving a balance between predictive accuracy and computational efficiency.

Data Caching: Leveraged Streamlit's @st.cache_data and @st.cache_resource decorators to intelligently cache fetched data and the trained model, drastically reducing load times on subsequent interactions.

Enhanced Real-Time Simulation: The "Live Stock Tracking" feature provides an interactive and dynamic price chart experience.

Methodology: Combines delayed intraday data fetching from yfinance with a stochastic simulation model to produce continuous price movements, creating a realistic live market feel.

Data-Informed Investment Tools: Provides powerful modules for comprehensive market analysis.

Sentiment Analysis: Fetches and analyzes recent news headlines using the NLTK VADER lexicon to gauge market sentiment (Positive, Negative, Neutral).

Portfolio Optimization: Suggests optimal stock weights for a given portfolio and risk tolerance, calculating key metrics such as expected annual return and Sharpe Ratio.

Professional UI/UX: The application features a clean, responsive user interface built on Streamlit, designed for a seamless and intuitive user experience.

🔐 Authentication & Data Storage
The project uses a custom, secure authentication system.

Database Schema: The application uses a file-based SQLite database named users.db to store user credentials.

Table Name: The table used for user accounts is named users.

Password Security: Passwords are not stored in plain text. Instead, they are securely hashed using the bcrypt library, ensuring that user data remains protected.

🚀 Getting Started
Follow these instructions to set up and run the application.

Prerequisites
Ensure you have Python 3.8 or a newer version installed on your system.

1. Clone the Repository
First, clone the project repository from GitHub to your local machine.

git clone [your_repository_url_here]
cd FutureStockAI


2. Install Dependencies
This project uses several key Python libraries. You can install all of them at once using the requirements.txt file.

pip install -r requirements.txt


Here are the libraries required for the project:

streamlit

pandas

numpy

plotly

yfinance

bcrypt

ta

tensorflow

nltk

3. Running the Application
To launch the web application, execute the following command in your terminal from the project's root directory:

streamlit run app.py


The application will automatically open in your default web browser.

📂 Project Structure
app.py: The main application orchestrator, managing the UI, session state, and user flow.

db_utils.py: Contains the core functions for database initialization and user authentication logic.

data_utils.py: Responsible for fetching, cleaning, and preprocessing data for both the prediction model and other modules.

model_trainer.py: Encapsulates the deep learning model (LSTM) building, training, and prediction logic.

sentiment_analyzer.py: Module for performing sentiment analysis on text data.

portfolio_optimizer.py: Handles the mathematical logic for portfolio optimization.

requirements.txt: Lists all project dependencies.

⚙️ Working
The application is built on a modular architecture, with each component performing a specific task.

Authentication: Upon launching the application, you will be directed to a login/signup interface.

Signup: To create a new account, go to the "Sign Up" tab, enter your details, and click the "Sign Up" button. A new entry will be securely added to the users.db database.

Login: After signing up, go to the "Login" tab, enter your registered email and password, and click the "Login" button. A successful login will set a session state variable, granting you access to the main dashboard.

Dashboard Access: Once logged in, you will be presented with a multi-tab dashboard.

Stock Prediction: To get a price forecast:

Navigate to the "Stock Prediction" tab.

On the sidebar, enter the stock ticker (e.g., AAPL) and adjust the number of days you want to predict.

Click the "Run Stock Prediction" button. The application will fetch the necessary data, train the LSTM model (this may take a moment on the first run), and display the predicted price chart and data table.

Real-Time Simulation: To view live market movements:

Go to the "Live Stock Tracking" tab.

Select a stock ticker from the dropdown menu (e.g., AAPL).

Set your desired refresh interval (in seconds) using the slider.

Click the "Start Live Tracking" button. The chart will begin to update with prices.

Data-Informed Decisions:

Sentiment Analysis: Go to this tab, enter a stock ticker, and click "Fetch & Analyze News Sentiment" to see recent news headlines and their sentiment scores.

Portfolio Optimization: In this tab, enter a list of stock tickers, set your risk tolerance, and click "Optimize Portfolio" to receive suggested asset weights and performance metrics.

Data Persistence: The application uses SQLite to ensure user credentials persist across application launches. All other data, such as prediction results or portfolio settings, is maintained using Streamlit's session state during a single user session.

📝 Important Notes
Live Data: For this project, a market price simulation is used for the real-time tracking feature. True, low-latency market data requires specialized APIs, which are typically subject to licensing fees.

News API: To enable the news sentiment analysis feature, a free API key from NewsAPI.org is required. You must insert your key into the sentiment_analyzer.py file.