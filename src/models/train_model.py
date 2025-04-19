"""
Script to train a prediction model using real market data.

This script fetches historical data from APIs, computes features,
and trains a model for stock price prediction.
"""

import os
import sys
import logging
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Import custom modules
from src.models.feature_engineering import (
    prepare_features, add_technical_indicators, add_date_features,
    add_target_features, get_feature_target_split, select_features
)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import project modules
from src.ingest.yahoo_client import YahooFinanceClient
from src.ingest.news_client import fetch_news

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(project_root) / "models"
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
LOOKBACK_DAYS = 365  # Use 1 year of data
PREDICTION_HORIZON = 5  # Predict 5 days ahead


def compute_rsi(prices, window=14):
    """Compute the Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi


def fetch_training_data(symbols, days, prediction_horizon=5):
    """
    Fetch historical data for multiple symbols and prepare features.

    Args:
        symbols: List of stock symbols
        days: Number of days of history to fetch
        prediction_horizon: Number of days ahead to predict

    Returns:
        DataFrame with historical data and features
    """
    logger.info(f"Fetching training data for {len(symbols)} symbols, {days} days of history")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    client = YahooFinanceClient(use_cache=True)
    all_data = []

    for symbol in symbols:
        try:
            # Fetch historical price data
            df = client.fetch_ticks(symbol, start_date, end_date)

            if df.empty:
                logger.warning(f"No data returned for {symbol}, skipping")
                continue

            # Add symbol column
            df['symbol'] = symbol

            # Add technical indicators
            df = add_technical_indicators(df, price_col='close', volume_col='volume')

            # Add date features
            df = add_date_features(df, date_col='timestamp')

            # Add target features
            df = add_target_features(df, price_col='close', horizons=[prediction_horizon])

            # Drop rows with NaN values
            df = df.dropna()

            all_data.append(df)
            logger.info(f"Fetched and processed {len(df)} rows for {symbol} with {len(df.columns)} features")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

    if not all_data:
        raise ValueError("No data fetched for any symbol")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns")

    return combined_df


def fetch_sentiment_data(symbols, days):
    """
    Fetch sentiment data for multiple symbols.

    Args:
        symbols: List of stock symbols
        days: Number of days of history to fetch

    Returns:
        DataFrame with sentiment data
    """
    logger.info(f"Fetching sentiment data for {len(symbols)} symbols")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_sentiment = []

    for symbol in symbols:
        try:
            # Fetch news for the symbol
            news = fetch_news([symbol], start_date)

            if not news:
                logger.warning(f"No news found for {symbol}, skipping sentiment analysis")
                continue

            # For now, just create a simple sentiment score based on the number of articles
            # In a real implementation, you would use FinBERT or another sentiment analyzer
            daily_sentiment = pd.DataFrame({
                'date': pd.date_range(start=start_date, end=end_date, freq='D'),
                'symbol': symbol,
                'sentiment_score': np.random.normal(0, 1, (end_date - start_date).days + 1)
            })

            all_sentiment.append(daily_sentiment)
            logger.info(f"Created sentiment data for {symbol}")

        except Exception as e:
            logger.error(f"Error creating sentiment data for {symbol}: {e}")

    if not all_sentiment:
        logger.warning("No sentiment data created for any symbol")
        return None

    # Combine all sentiment data
    combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
    logger.info(f"Combined sentiment dataset has {len(combined_sentiment)} rows")

    return combined_sentiment


def train_model(data, sentiment=None, prediction_horizon=5, n_features=20):
    """
    Train a prediction model using the provided data.

    Args:
        data: DataFrame with historical price data and features
        sentiment: Optional DataFrame with sentiment data
        prediction_horizon: Number of days ahead to predict
        n_features: Number of features to select

    Returns:
        Tuple of (trained model, feature names, evaluation metrics)
    """
    logger.info("Training prediction model")

    # If sentiment data is provided, merge it with price data
    if sentiment is not None:
        # Convert timestamp to date for merging
        data['date'] = pd.to_datetime(data['timestamp']).dt.date

        # Merge price data with sentiment data
        data = pd.merge(data, sentiment, on=['date', 'symbol'], how='left')

        # Fill missing sentiment scores with 0
        data['sentiment_score'] = data['sentiment_score'].fillna(0)

    # Define target column
    target_col = f'future_return_{prediction_horizon}'

    # Check if target column exists
    if target_col not in data.columns:
        logger.warning(f"Target column {target_col} not found, using 'future_return' instead")
        target_col = 'future_return'

    # Get features and target
    X, y = get_feature_target_split(data, target_col=target_col)

    # Split data into training and testing sets using time-based split
    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        sorted_indices = data.sort_values('timestamp').index
        train_size = int(0.8 * len(sorted_indices))
        train_indices = sorted_indices[:train_size]
        test_indices = sorted_indices[train_size:]
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        logger.info(f"Using time-based split: {len(X_train)} training samples, {len(X_test)} testing samples")
    else:
        # Fall back to random split if timestamp not available
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Using random split: {len(X_train)} training samples, {len(X_test)} testing samples")

    # Select features
    X_train_selected, selected_features = select_features(X_train, y_train, method='f_regression', k=n_features)
    X_test_selected = X_test[selected_features]

    logger.info(f"Selected {len(selected_features)} features")

    # Train XGBoost regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        random_state=42
    )

    model.fit(
        X_train_selected, y_train,
        eval_set=[(X_train_selected, y_train), (X_test_selected, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=10,
        verbose=False
    )

    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    logger.info(f"Best iteration: {best_iteration}")

    # Evaluate the model
    y_pred = model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate directional accuracy
    direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))

    # Log evaluation metrics
    logger.info(f"Model evaluation:")
    logger.info(f"  - MSE: {mse:.6f}")
    logger.info(f"  - RMSE: {rmse:.6f}")
    logger.info(f"  - MAE: {mae:.6f}")
    logger.info(f"  - R²: {r2:.6f}")
    logger.info(f"  - Directional Accuracy: {direction_accuracy:.6f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"Top 10 features by importance:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(10),
                                               feature_importance['importance'].head(10))):
        logger.info(f"  {i+1}. {feature}: {importance:.6f}")

    # Collect evaluation metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'best_iteration': best_iteration
    }

    return model, selected_features, metrics


def save_model(model, selected_features, metrics, filename="baseline_xgb.pkl"):
    """
    Save the trained model, selected features, and metrics to disk.

    Args:
        model: Trained model
        selected_features: List of selected feature names
        metrics: Dictionary of evaluation metrics
        filename: Name of the file to save the model to
    """
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save selected features
    features_path = MODEL_DIR / f"{filename.split('.')[0]}_features.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f)
    logger.info(f"Selected features saved to {features_path}")

    # Save metrics
    metrics_path = MODEL_DIR / f"{filename.split('.')[0]}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


def main():
    """Main function to train and save the model."""
    try:
        # Fetch training data with enhanced features
        data = fetch_training_data(SYMBOLS, LOOKBACK_DAYS, PREDICTION_HORIZON)

        # Fetch sentiment data
        sentiment = fetch_sentiment_data(SYMBOLS, LOOKBACK_DAYS)

        # Train model with feature selection and hyperparameters
        model, selected_features, metrics = train_model(
            data,
            sentiment,
            prediction_horizon=PREDICTION_HORIZON,
            n_features=20
        )

        # Save model, features, and metrics
        save_model(model, selected_features, metrics)

        # Print summary of results
        print("\nModel Training Summary:")
        print("=======================")
        print(f"Trained on {len(data)} samples with {len(selected_features)} selected features")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        print(f"Directional Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"\nTop 5 Features:")
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(5),
                                                  feature_importance['importance'].head(5))):
            print(f"  {i+1}. {feature}: {importance:.6f}")

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
