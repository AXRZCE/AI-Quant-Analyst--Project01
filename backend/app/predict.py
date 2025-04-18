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
from src.ingest.news_client import fetch_news
from .models import PredictionResponse

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
    model_path = Path("/models/baseline_xgb.pkl")

    # If model doesn't exist, create a dummy model
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, using dummy model")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10)
        model.fit(
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0.01, -0.01, 0.02])
        )
        return model

    return joblib.load(model_path)

def run_prediction(symbol: str, days: int) -> PredictionResponse:
    """Run the prediction pipeline for a given symbol"""
    logger.info(f"Running prediction for {symbol} with {days} days of history")
    now = datetime.now(timezone.utc)

    try:
        # 1. Fetch historical ticks
        client = PolygonClient()
        start_date = now - timedelta(days=days)
        df = client.fetch_ticks(symbol, start_date, now)

        # If no data, generate dummy data
        if df.empty:
            logger.warning(f"No data returned for {symbol}, generating dummy data")
            dates = pd.date_range(start=start_date, end=now, freq='1H')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.normal(100, 5, len(dates)),
                'high': np.random.normal(105, 5, len(dates)),
                'low': np.random.normal(95, 5, len(dates)),
                'close': np.random.normal(100, 5, len(dates)),
                'volume': np.random.randint(1000, 10000, len(dates))
            }).set_index('timestamp')

        # 2. Compute features
        df["ma_5"] = df["close"].rolling(5).mean()
        df["rsi_14"] = compute_rsi(df["close"], 14)
        df = df.dropna()

        # 3. Fetch recent news & sentiment
        try:
            news = fetch_news([symbol], now - timedelta(hours=24))
            texts = [n["title"] for n in news] or ["No recent news for " + symbol]
            sentiment_analyzer = get_sentiment_analyzer()
            sent = sentiment_analyzer.analyze(texts)
            avg = {k: sum(d[k] for d in sent) / len(sent) for k in sent[0]}
        except Exception as e:
            logger.error(f"Error fetching news/sentiment: {e}")
            avg = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

        # 4. Model inference
        features = pd.DataFrame([{
            "ma_5": df["ma_5"].iloc[-1],
            "rsi_14": df["rsi_14"].iloc[-1],
            "sentiment_positive": avg["positive"],
            "sentiment_neutral": avg["neutral"],
            "sentiment_negative": avg["negative"]
        }])

        model = load_model()
        pred = float(model.predict(features)[0])

        # 5. Build response
        return PredictionResponse(
            timestamps=[ts.isoformat() for ts in df.index],
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
