# sentiment_analyzer.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import requests # NEW IMPORT for making HTTP requests
import datetime # NEW IMPORT for date calculations

# --- Configuration ---
# IMPORTANT: Replace 'YOUR_NEWSAPI_KEY' with your actual API key from NewsAPI.org
NEWSAPI_KEY = "1c4c338a6dc54f17beac7d3ec030a7d0"
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def get_sentiment_analyzer():
    """
    Initializes and returns the VADER Sentiment Intensity Analyzer.
    Uses st.cache_resource to ensure it's loaded only once.
    """
    return SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using VADER.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing 'neg', 'neu', 'pos', and 'compound' scores.
              Compound score ranges from -1 (most negative) to +1 (most positive).
    """
    analyzer = get_sentiment_analyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

def get_sentiment_label(compound_score: float) -> str:
    """
    Converts a compound sentiment score into a categorical label.

    Args:
        compound_score (float): The compound sentiment score from VADER.

    Returns:
        str: 'Positive', 'Negative', or 'Neutral'.
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@st.cache_data(ttl=3600) # Cache news data for 1 hour to avoid hitting API limits
def fetch_news_headlines(query: str, api_key: str, days_back: int = 7) -> list[dict]:
    """
    Fetches recent news headlines related to a query using NewsAPI.org.

    Args:
        query (str): The search query (e.g., stock ticker or company name).
        api_key (str): Your NewsAPI.org API key.
        days_back (int): How many days back to search for news.

    Returns:
        list[dict]: A list of dictionaries, each representing a news article.
    """
    if not api_key or api_key == "YOUR_NEWSAPI_KEY":
        st.error("NewsAPI Key is missing or not set. Please get a key from NewsAPI.org and update 'sentiment_analyzer.py'.")
        return []

    from_date = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.date.today().strftime('%Y-%m-%d')

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "from": from_date,
        "to": to_date,
        "apiKey": api_key
    }

    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        articles = response.json().get('articles', [])
        return articles
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error fetching news: {http_err}. Status Code: {response.status_code}. Response: {response.text}")
        if response.status_code == 401:
            st.error("Your NewsAPI key might be invalid or missing. Please check it.")
        elif response.status_code == 429:
            st.error("NewsAPI rate limit exceeded. Please wait a moment before trying again.")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error fetching news: {conn_err}. Please check your internet connection.")
        return []
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error fetching news: {timeout_err}. The request took too long.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching news: {e}")
        return []

