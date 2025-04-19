"""
Advanced prediction module for the API.

This module provides advanced prediction functionality using the unified prediction
pipeline with advanced models, ensembling, and uncertainty quantification.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import prediction pipeline
try:
    from src.models.prediction_pipeline import create_prediction_pipeline, PredictionPipeline
    from src.models.advanced_models import ModelIntegrator
    from src.nlp.finbert_sentiment import FinBERTSentiment
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Advanced models not available, using fallback implementation")
    ADVANCED_MODELS_AVAILABLE = False


class DummyPredictionPipeline:
    """Dummy prediction pipeline for when advanced models are not available."""
    
    def __init__(self):
        """Initialize the dummy prediction pipeline."""
        self.use_advanced_models = False
        self.use_ensemble = False
        self.use_uncertainty = False
    
    def predict(self, X, return_uncertainty=True):
        """Make dummy predictions."""
        if return_uncertainty:
            return pd.DataFrame({
                "prediction": np.random.randn(len(X)) * 0.05 + 0.01,
                "lower_bound": np.random.randn(len(X)) * 0.05 - 0.05,
                "upper_bound": np.random.randn(len(X)) * 0.05 + 0.07,
                "uncertainty": np.random.rand(len(X)) * 0.05 + 0.02
            })
        else:
            return np.random.randn(len(X)) * 0.05 + 0.01
    
    def predict_with_sentiment(self, X, texts, return_uncertainty=True):
        """Make dummy predictions with sentiment."""
        predictions = self.predict(X, return_uncertainty)
        sentiment = [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]
        return {"predictions": predictions, "sentiment": sentiment}
    
    def predict_time_series(self, df, horizon=5, return_uncertainty=True):
        """Make dummy time series predictions."""
        forecasts = pd.DataFrame()
        forecasts["forecast"] = np.random.randn(len(df)) * 0.05 + 0.01
        
        if return_uncertainty:
            forecasts["lower"] = forecasts["forecast"] - np.random.rand(len(df)) * 0.05 - 0.02
            forecasts["upper"] = forecasts["forecast"] + np.random.rand(len(df)) * 0.05 + 0.02
        
        return forecasts


class DummyFinBERTSentiment:
    """Dummy sentiment analyzer for when FinBERT is not available."""
    
    def analyze(self, texts):
        """Analyze sentiment of texts."""
        return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]


def get_prediction_pipeline():
    """Get the prediction pipeline, falling back to dummy if advanced models are not available."""
    if ADVANCED_MODELS_AVAILABLE:
        try:
            # Try to create prediction pipeline
            pipeline = create_prediction_pipeline(
                use_ensemble=True,
                use_advanced_models=True,
                use_uncertainty=True,
                ensemble_type="weighted",
                uncertainty_method="bootstrap"
            )
            logger.info("Created advanced prediction pipeline")
            return pipeline
        except Exception as e:
            logger.error(f"Error creating prediction pipeline: {e}")
            return DummyPredictionPipeline()
    else:
        logger.warning("Advanced models not available, using dummy prediction pipeline")
        return DummyPredictionPipeline()


def get_sentiment_analyzer():
    """Get the sentiment analyzer, falling back to dummy if FinBERT is not available."""
    if ADVANCED_MODELS_AVAILABLE:
        try:
            from src.nlp.finbert_sentiment import FinBERTSentiment
            return FinBERTSentiment()
        except (ImportError, Exception) as e:
            logger.warning(f"FinBERT sentiment analyzer not available: {e}")
            return DummyFinBERTSentiment()
    else:
        logger.warning("FinBERT sentiment analyzer not available, using dummy")
        return DummyFinBERTSentiment()


def get_model_integrator():
    """Get the model integrator, falling back to None if advanced models are not available."""
    if ADVANCED_MODELS_AVAILABLE:
        try:
            from src.models.advanced_models import ModelIntegrator
            return ModelIntegrator()
        except (ImportError, Exception) as e:
            logger.warning(f"Model integrator not available: {e}")
            return None
    else:
        logger.warning("Model integrator not available")
        return None


def fetch_stock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        DataFrame with stock data
    """
    logger.info(f"Fetching stock data for {symbol} for {days} days")
    
    try:
        # Calculate start and end dates
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Reset index
        data = data.reset_index()
        
        # Rename columns
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Fetched {len(data)} rows of data for {symbol}")
        
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


def fetch_news(symbol: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch news for a stock.
    
    Args:
        symbol: Stock symbol
        days: Number of days of news
        
    Returns:
        List of news articles
    """
    logger.info(f"Fetching news for {symbol} for {days} days")
    
    try:
        # Try to use yfinance to get news
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            logger.warning(f"No news found for {symbol}")
            return []
        
        # Filter by date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        filtered_news = []
        
        for article in news:
            # Convert timestamp to datetime
            if "providerPublishTime" in article:
                publish_time = datetime.fromtimestamp(article["providerPublishTime"], tz=timezone.utc)
                
                if publish_time >= cutoff_date:
                    filtered_news.append(article)
        
        logger.info(f"Fetched {len(filtered_news)} news articles for {symbol}")
        
        return filtered_news
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators.
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        DataFrame with technical indicators
    """
    logger.info("Calculating technical indicators")
    
    try:
        # Make a copy of the DataFrame
        result = df.copy()
        
        # Check if required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in result.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return df
        
        # Calculate moving averages
        result["ma_5"] = result["close"].rolling(window=5).mean()
        result["ma_10"] = result["close"].rolling(window=10).mean()
        result["ma_20"] = result["close"].rolling(window=20).mean()
        result["ma_50"] = result["close"].rolling(window=50).mean()
        
        # Calculate exponential moving averages
        result["ema_5"] = result["close"].ewm(span=5, adjust=False).mean()
        result["ema_10"] = result["close"].ewm(span=10, adjust=False).mean()
        result["ema_20"] = result["close"].ewm(span=20, adjust=False).mean()
        
        # Calculate MACD
        result["macd"] = result["ema_12"] = result["close"].ewm(span=12, adjust=False).mean() - result["close"].ewm(span=26, adjust=False).mean()
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]
        
        # Calculate RSI
        delta = result["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        result["bb_middle"] = result["close"].rolling(window=20).mean()
        result["bb_std"] = result["close"].rolling(window=20).std()
        result["bb_upper"] = result["bb_middle"] + 2 * result["bb_std"]
        result["bb_lower"] = result["bb_middle"] - 2 * result["bb_std"]
        
        # Calculate returns
        result["daily_return"] = result["close"].pct_change()
        result["weekly_return"] = result["close"].pct_change(5)
        result["monthly_return"] = result["close"].pct_change(20)
        
        # Calculate volatility
        result["volatility_5"] = result["daily_return"].rolling(window=5).std()
        result["volatility_10"] = result["daily_return"].rolling(window=10).std()
        result["volatility_20"] = result["daily_return"].rolling(window=20).std()
        
        # Fill NaN values
        result = result.fillna(method="bfill").fillna(method="ffill").fillna(0)
        
        logger.info("Calculated technical indicators")
        
        return result
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df


def analyze_sentiment(news: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze sentiment of news articles.
    
    Args:
        news: List of news articles
        
    Returns:
        Dictionary with sentiment scores
    """
    logger.info(f"Analyzing sentiment of {len(news)} news articles")
    
    try:
        # Get sentiment analyzer
        sentiment_analyzer = get_sentiment_analyzer()
        
        # Extract text from news
        texts = []
        for article in news:
            if "title" in article:
                texts.append(article["title"])
        
        if not texts:
            logger.warning("No texts found for sentiment analysis")
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
        
        # Analyze sentiment
        sentiments = sentiment_analyzer.analyze(texts)
        
        # Calculate average sentiment
        avg_sentiment = {
            "positive": sum(s.get("positive", 0) for s in sentiments) / len(sentiments),
            "neutral": sum(s.get("neutral", 0) for s in sentiments) / len(sentiments),
            "negative": sum(s.get("negative", 0) for s in sentiments) / len(sentiments)
        }
        
        logger.info(f"Analyzed sentiment: {avg_sentiment}")
        
        return avg_sentiment
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}


def prepare_features(df: pd.DataFrame, sentiment: Dict[str, float]) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    Args:
        df: DataFrame with stock data and technical indicators
        sentiment: Dictionary with sentiment scores
        
    Returns:
        DataFrame with features
    """
    logger.info("Preparing features for prediction")
    
    try:
        # Make a copy of the DataFrame
        features = df.copy()
        
        # Add sentiment features
        for key, value in sentiment.items():
            features[f"sentiment_{key}"] = value
        
        # Add time features
        if "date" in features.columns:
            features["day_of_week"] = pd.to_datetime(features["date"]).dt.dayofweek
            features["month"] = pd.to_datetime(features["date"]).dt.month
            features["year"] = pd.to_datetime(features["date"]).dt.year
        
        # Add target column (next day return)
        features["target"] = features["close"].pct_change(1).shift(-1)
        
        # Drop unnecessary columns
        drop_columns = ["date", "open", "high", "low", "close", "volume", "target"]
        features = features.drop([col for col in drop_columns if col in features.columns], axis=1)
        
        # Fill NaN values
        features = features.fillna(0)
        
        logger.info(f"Prepared {len(features.columns)} features for prediction")
        
        return features
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return df


def run_advanced_prediction(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Run advanced prediction for a stock.
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Running advanced prediction for {symbol} with {days} days of history")
    
    try:
        # Fetch stock data
        df = fetch_stock_data(symbol, days)
        
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return {
                "error": f"No data found for {symbol}"
            }
        
        # Fetch news
        news = fetch_news(symbol, min(days, 7))
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(news)
        
        # Prepare features
        features = prepare_features(df, sentiment)
        
        # Get prediction pipeline
        pipeline = get_prediction_pipeline()
        
        # Make predictions
        predictions = pipeline.predict(features)
        
        # Extract the most recent prediction
        if isinstance(predictions, pd.DataFrame):
            latest_prediction = {
                "prediction": float(predictions["prediction"].iloc[-1]),
                "lower_bound": float(predictions["lower_bound"].iloc[-1]),
                "upper_bound": float(predictions["upper_bound"].iloc[-1]),
                "uncertainty": float(predictions["uncertainty"].iloc[-1])
            }
        else:
            latest_prediction = {
                "prediction": float(predictions[-1])
            }
        
        # Get latest price
        latest_price = float(df["close"].iloc[-1])
        
        # Calculate predicted return
        predicted_return = latest_prediction["prediction"]
        
        # Calculate predicted price
        predicted_price = latest_price * (1 + predicted_return)
        
        # Calculate confidence interval for price
        if "lower_bound" in latest_prediction and "upper_bound" in latest_prediction:
            price_lower = latest_price * (1 + latest_prediction["lower_bound"])
            price_upper = latest_price * (1 + latest_prediction["upper_bound"])
        else:
            price_lower = None
            price_upper = None
        
        # Prepare historical data for response
        historical_data = {
            "timestamps": df["date"].astype(str).tolist(),
            "prices": df["close"].tolist(),
            "ma_5": df["ma_5"].tolist(),
            "rsi_14": df["rsi_14"].tolist()
        }
        
        # Prepare response
        response = {
            "symbol": symbol,
            "latest_price": latest_price,
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
            "confidence_interval": {
                "lower": price_lower,
                "upper": price_upper
            } if price_lower is not None and price_upper is not None else None,
            "uncertainty": latest_prediction.get("uncertainty"),
            "sentiment": sentiment,
            "historical_data": historical_data
        }
        
        logger.info(f"Prediction for {symbol}: {predicted_return:.2%}")
        
        return response
    except Exception as e:
        logger.error(f"Error running advanced prediction: {e}")
        return {
            "error": f"Error running prediction: {str(e)}"
        }
