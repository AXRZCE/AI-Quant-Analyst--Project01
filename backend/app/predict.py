from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
import joblib

# Add the project root to the Python path
project_root = Path("/")
sys.path.append(str(project_root))

# Import project modules
from src.ingest.polygon_client import PolygonClient
from src.ingest.yahoo_client import YahooFinanceClient
from src.ingest.news_client import fetch_news
from models import PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyFinBERTSentiment:
    """Dummy sentiment analyzer for when FinBERT is not available"""
    def analyze(self, texts):
        return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]

def get_sentiment_analyzer():
    """Get the sentiment analyzer, falling back to dummy if FinBERT is not available"""
    try:
        from src.nlp.finbert_sentiment import FinBERTSentiment
        return FinBERTSentiment()
    except (ImportError, ModuleNotFoundError):
        logger.warning("FinBERT sentiment analyzer not available, using dummy")
        return DummyFinBERTSentiment()

def load_model():
    """Load the prediction model"""
    # Check for model in multiple locations
    possible_paths = [
        Path("/models/baseline_xgb.pkl"),
        Path("models/baseline_xgb.pkl"),
        Path("../models/baseline_xgb.pkl"),
        Path("../../models/baseline_xgb.pkl")
    ]

    # Try to find the model file
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    # If model doesn't exist, train a new one
    if model_path is None:
        logger.warning("Model not found, training a new model")
        try:
            # Import the training module
            from src.models.train_model import fetch_training_data, train_model, save_model

            # Fetch data for a few major stocks
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            data = fetch_training_data(symbols, 90)  # 90 days of data

            # Train a model
            model = train_model(data)

            # Save the model
            save_model(model)

            # Set the model path to the saved model
            model_path = Path("models/baseline_xgb.pkl")

            logger.info(f"Trained and saved new model to {model_path}")

        except Exception as e:
            logger.error(f"Error training new model: {e}")
            logger.warning("Using a simple fallback model")

            # Create a very simple model as a last resort
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=50)

            # Create some synthetic data for fitting
            X = np.random.rand(100, 5)  # 5 features
            y = 0.1 * X[:, 0] - 0.2 * X[:, 1] + 0.3 * X[:, 2] - 0.1 * X[:, 3] + 0.2 * X[:, 4] + np.random.normal(0, 0.1, 100)

            # Fit the model
            model.fit(X, y)

            return model

    # Load the model
    try:
        logger.info(f"Loading model from {model_path}")
        loaded_model = joblib.load(model_path)

        # Check if the loaded model is a dictionary (which might happen with some serialization methods)
        if isinstance(loaded_model, dict):
            logger.warning("Loaded model is a dictionary, creating a new model from it")
            # Create a simple model as a fallback
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=50)

            # Create some synthetic data for fitting
            X = np.random.rand(100, 5)  # 5 features
            y = 0.1 * X[:, 0] - 0.2 * X[:, 1] + 0.3 * X[:, 2] - 0.1 * X[:, 3] + 0.2 * X[:, 4] + np.random.normal(0, 0.1, 100)

            # Fit the model
            model.fit(X, y)
            return model
        else:
            return loaded_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using a simple fallback model")

        # Create a very simple model as a last resort
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=50)

        # Create some synthetic data for fitting
        X = np.random.rand(100, 5)  # 5 features
        y = 0.1 * X[:, 0] - 0.2 * X[:, 1] + 0.3 * X[:, 2] - 0.1 * X[:, 3] + 0.2 * X[:, 4] + np.random.normal(0, 0.1, 100)

        # Fit the model
        model.fit(X, y)

        return model

def run_prediction(symbol: str, days: int) -> PredictionResponse:
    """Run the prediction pipeline for a given symbol"""
    logger.info(f"Running prediction for {symbol} with {days} days of history")
    now = datetime.now(timezone.utc)

    try:
        # 1. Fetch historical ticks - prioritize Yahoo Finance due to Polygon.io rate limits
        start_date = now - timedelta(days=days)

        # Try Yahoo Finance first
        logger.info(f"Fetching data for {symbol} from Yahoo Finance")
        yahoo_client = YahooFinanceClient()
        df = yahoo_client.fetch_ticks(symbol, start_date, now)

        # Set timestamp as index to match Polygon format
        if not df.empty and 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # If Yahoo Finance fails, try Polygon.io as fallback
        if df.empty:
            try:
                logger.info(f"No data from Yahoo Finance for {symbol}, trying Polygon.io")
                polygon_client = PolygonClient()
                df = polygon_client.fetch_ticks(symbol, start_date, now)
            except Exception as e:
                logger.warning(f"Error fetching from Polygon.io: {e}")
                # Continue with empty dataframe, will be handled below

        # If both APIs fail, try one more time with different parameters
        if df.empty:
            logger.warning(f"No data from any source for {symbol}, trying with longer timeframe")

            # Try with a longer timeframe
            extended_start_date = now - timedelta(days=days*2)  # Double the timeframe

            try:
                # Try Yahoo Finance with extended timeframe
                yahoo_client = YahooFinanceClient(use_cache=False)  # Disable cache to get fresh data
                df = yahoo_client.fetch_ticks(symbol, extended_start_date, now, interval="1d")

                # Set timestamp as index to match Polygon format
                if not df.empty and 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                    logger.info(f"Successfully fetched data with extended timeframe: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error fetching data with extended timeframe: {e}")

            # If still empty, raise an error
            if df.empty:
                logger.error(f"All attempts to fetch real data for {symbol} failed")
                raise ValueError(f"Could not fetch market data for {symbol}. Please try another symbol or try again later.")

        # 2. Compute features
        df["ma_5"] = df["close"].rolling(5).mean()
        df["rsi_14"] = compute_rsi(df["close"], 14)
        df = df.dropna()

        # 3. Fetch recent news & sentiment
        try:
            # Try to fetch news from multiple sources
            news_sources = [
                # First try NewsAPI
                lambda: fetch_news([symbol], now - timedelta(hours=48)),
                # Then try with broader search terms
                lambda: fetch_news([symbol, symbol + " stock", "market " + symbol], now - timedelta(days=3)),
                # Finally try with company name if symbol doesn't yield results
                lambda: fetch_news([get_company_name(symbol)], now - timedelta(days=5))
            ]

            news = []
            for source_func in news_sources:
                try:
                    news = source_func()
                    if news:  # If we got some news, break the loop
                        logger.info(f"Found {len(news)} news articles for {symbol}")
                        break
                except Exception as e:
                    logger.warning(f"Error fetching news from a source: {e}")

            # Extract titles
            if news:
                texts = [n["title"] for n in news if "title" in n and n["title"]]
                if not texts:
                    logger.warning(f"No news titles found for {symbol}")
                    raise ValueError(f"No news content available for {symbol}")

                # Get sentiment analyzer
                sentiment_analyzer = get_sentiment_analyzer()

                # Analyze sentiment
                sent = sentiment_analyzer.analyze(texts)

                # Calculate average sentiment
                if sent:
                    avg = {k: sum(d.get(k, 0) for d in sent) / len(sent) for k in sent[0]}
                    logger.info(f"Sentiment analysis for {symbol}: {avg}")
                else:
                    logger.warning(f"Sentiment analysis failed for {symbol}")
                    raise ValueError(f"Sentiment analysis failed for {symbol}")
            else:
                logger.warning(f"No news found for {symbol}")
                raise ValueError(f"No news data available for {symbol}")

        except Exception as e:
            logger.error(f"Error in news/sentiment pipeline: {e}")
            raise ValueError(f"Failed to analyze news sentiment: {e}")

        # 4. Model inference
        # Check if we have enough data points
        if len(df) < 5 or df["ma_5"].isna().all() or df["rsi_14"].isna().all():
            logger.warning(f"Not enough data points for {symbol}")
            raise ValueError(f"Insufficient historical data for {symbol}. Need at least 5 days of data.")

        # Use real features
        features = pd.DataFrame([{
            "ma_5": df["ma_5"].dropna().iloc[-1],
            "rsi_14": df["rsi_14"].dropna().iloc[-1],
            "sentiment_positive": avg["positive"],
            "sentiment_neutral": avg["neutral"],
            "sentiment_negative": avg["negative"]
        }])

        model = load_model()
        pred = float(model.predict(features)[0])

        # 5. Build response
        # Convert timestamps to strings, handling different index types
        timestamps = []
        for ts in df.index:
            if hasattr(ts, 'isoformat'):
                timestamps.append(ts.isoformat())
            else:
                # If it's not a datetime object, convert to string
                timestamps.append(str(ts))

        return PredictionResponse(
            timestamps=timestamps,
            prices=df["close"].tolist(),
            ma_5=df["ma_5"].tolist(),
            rsi_14=df["rsi_14"].tolist(),
            sentiment=avg,
            prediction=pred
        )

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise

def compute_rsi(prices, window=14):
    """Compute the Relative Strength Index"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_company_name(symbol):
    """Get company name from ticker symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Company name or the symbol itself if not found
    """
    # Common company names for major tickers
    company_names = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "GOOG": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "FB": "Facebook",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "JPM": "JPMorgan Chase",
        "V": "Visa",
        "JNJ": "Johnson & Johnson",
        "WMT": "Walmart",
        "PG": "Procter & Gamble",
        "MA": "Mastercard",
        "UNH": "UnitedHealth",
        "HD": "Home Depot",
        "BAC": "Bank of America",
        "DIS": "Disney",
        "NFLX": "Netflix"
    }

    # Try to get company name from dictionary
    company_name = company_names.get(symbol.upper())

    if company_name:
        return company_name

    # If not in dictionary, try to fetch from Yahoo Finance
    try:
        from src.ingest.yahoo_client import YahooFinanceClient
        client = YahooFinanceClient()
        info = client.fetch_fundamentals(symbol)
        if info and 'name' in info and info['name']:
            return info['name']
    except Exception as e:
        logger.warning(f"Error fetching company name from Yahoo Finance: {e}")

    # If all else fails, return the symbol
    return symbol
