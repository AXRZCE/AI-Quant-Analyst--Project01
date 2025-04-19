"""
Example script demonstrating the advanced models integration.

This script shows how to use the advanced models integration with TFT, FinBERT,
model ensembling, and uncertainty quantification.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.prediction_pipeline import create_prediction_pipeline
from src.models.advanced_models import ModelIntegrator, TFTWrapper, FinBERTWrapper
from src.models.ensemble import create_ensemble, WeightedEnsemble
from src.models.uncertainty import create_uncertainty_quantifier, BootstrapUncertainty
from src.models.model_selection import ModelRegistry, ModelSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbol, days=365):
    """Fetch stock data from Yahoo Finance."""
    try:
        import yfinance as yf
        
        # Calculate start and end dates
        end_date = datetime.now()
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
            return None
        
        # Reset index
        data = data.reset_index()
        
        # Rename columns
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Fetched {len(data)} rows of data for {symbol}")
        
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return None


def calculate_features(df):
    """Calculate technical indicators and features."""
    try:
        # Make a copy of the DataFrame
        result = df.copy()
        
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
        result["ema_12"] = result["close"].ewm(span=12, adjust=False).mean()
        result["ema_26"] = result["close"].ewm(span=26, adjust=False).mean()
        result["macd"] = result["ema_12"] - result["ema_26"]
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
        
        # Add target column (next day return)
        result["target"] = result["close"].pct_change(1).shift(-1)
        
        # Fill NaN values
        result = result.fillna(method="bfill").fillna(method="ffill").fillna(0)
        
        logger.info(f"Calculated {len(result.columns)} features")
        
        return result
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        return df


def prepare_data(symbol, days=365):
    """Prepare data for modeling."""
    # Fetch data
    df = fetch_stock_data(symbol, days)
    
    if df is None or df.empty:
        logger.error(f"No data available for {symbol}")
        return None, None
    
    # Calculate features
    df = calculate_features(df)
    
    # Split into features and target
    X = df.drop(["target", "date"], axis=1, errors="ignore")
    y = df["target"]
    
    return X, y


def demo_finbert():
    """Demonstrate FinBERT sentiment analysis."""
    logger.info("Demonstrating FinBERT sentiment analysis")
    
    # Create FinBERT wrapper
    finbert = FinBERTWrapper()
    
    # Example texts
    texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "The stock price plummeted after the CEO resigned amid accounting irregularities.",
        "The market remained flat as investors awaited the Fed's decision on interest rates.",
        "The company announced a new product line that could significantly increase revenue.",
        "Analysts are concerned about the company's high debt levels and declining margins."
    ]
    
    # Analyze sentiment
    sentiments = finbert.analyze(texts)
    
    # Print results
    print("\nFinBERT Sentiment Analysis:")
    print("=" * 50)
    
    for i, (text, sentiment) in enumerate(zip(texts, sentiments)):
        print(f"\nText {i+1}: {text}")
        print(f"Sentiment: Positive={sentiment.get('positive', 0):.2f}, "
              f"Neutral={sentiment.get('neutral', 0):.2f}, "
              f"Negative={sentiment.get('negative', 0):.2f}")
    
    # Get sentiment features
    sentiment_df = finbert.get_sentiment_features(texts)
    
    print("\nSentiment Features:")
    print(sentiment_df)


def demo_tft(symbol="AAPL"):
    """Demonstrate Temporal Fusion Transformer."""
    logger.info(f"Demonstrating TFT for {symbol}")
    
    # Prepare data
    X, y = prepare_data(symbol)
    
    if X is None or y is None:
        return
    
    # Create TFT wrapper
    tft = TFTWrapper()
    
    # Add time index column
    X["time_idx"] = np.arange(len(X))
    
    # Add symbol column
    X["symbol"] = symbol
    
    # Add target column
    X["target"] = y
    
    # Prepare data for TFT
    tft.prepare_data(X)
    
    # Make predictions
    predictions = tft.predict(X)
    
    # Get prediction intervals
    lower, upper = tft.predict_interval(X)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y, 'b-', label='Actual')
    plt.plot(predictions, 'r-', label='Prediction')
    plt.fill_between(
        np.arange(len(predictions)),
        lower, upper,
        alpha=0.2, color='r',
        label='95% Confidence Interval'
    )
    plt.title(f'TFT Predictions for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/tft_{symbol}.png")
    
    print(f"\nTFT predictions saved to results/plots/tft_{symbol}.png")


def demo_ensemble(symbol="AAPL"):
    """Demonstrate model ensembling."""
    logger.info(f"Demonstrating model ensembling for {symbol}")
    
    # Prepare data
    X, y = prepare_data(symbol)
    
    if X is None or y is None:
        return
    
    # Split data into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Create base models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    
    models = [
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("lr", LinearRegression()),
        ("xgb", XGBRegressor(n_estimators=100, random_state=42))
    ]
    
    # Fit base models
    for name, model in models:
        model.fit(X_train, y_train)
    
    # Create ensemble
    ensemble = create_ensemble(models, ensemble_type="weighted", optimize_weights=True)
    
    # Fit ensemble
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    y_pred = ensemble.predict(X_test)
    
    # Get prediction intervals
    lower, upper = ensemble.predict_interval(X_test)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, 'b-', label='Actual')
    plt.plot(y_pred, 'r-', label='Ensemble Prediction')
    plt.fill_between(
        np.arange(len(y_pred)),
        lower, upper,
        alpha=0.2, color='r',
        label='95% Confidence Interval'
    )
    plt.title(f'Ensemble Predictions for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/ensemble_{symbol}.png")
    
    print(f"\nEnsemble predictions saved to results/plots/ensemble_{symbol}.png")
    
    # Print model importances
    importances = ensemble.get_model_importances()
    
    print("\nModel Importances:")
    for name, importance in importances.items():
        print(f"{name}: {importance:.4f}")


def demo_uncertainty(symbol="AAPL"):
    """Demonstrate uncertainty quantification."""
    logger.info(f"Demonstrating uncertainty quantification for {symbol}")
    
    # Prepare data
    X, y = prepare_data(symbol)
    
    if X is None or y is None:
        return
    
    # Split data into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Create base model
    from xgboost import XGBRegressor
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    
    # Create uncertainty quantifier
    uncertainty = create_uncertainty_quantifier(
        model,
        method="bootstrap",
        n_estimators=100,
        subsample=0.8,
        random_state=42
    )
    
    # Fit uncertainty quantifier
    uncertainty.fit(X_train, y_train)
    
    # Make predictions
    y_pred = uncertainty.predict(X_test)
    
    # Get prediction intervals
    lower, upper = uncertainty.predict_interval(X_test)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, 'b-', label='Actual')
    plt.plot(y_pred, 'r-', label='Prediction')
    plt.fill_between(
        np.arange(len(y_pred)),
        lower, upper,
        alpha=0.2, color='r',
        label='95% Confidence Interval'
    )
    plt.title(f'Uncertainty Quantification for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/uncertainty_{symbol}.png")
    
    print(f"\nUncertainty quantification saved to results/plots/uncertainty_{symbol}.png")
    
    # Calculate coverage
    coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
    
    print(f"\nConfidence Interval Coverage: {coverage:.2%}")
    
    # Calculate average interval width
    avg_width = np.mean(upper - lower)
    
    print(f"Average Interval Width: {avg_width:.4f}")


def demo_prediction_pipeline(symbol="AAPL"):
    """Demonstrate the unified prediction pipeline."""
    logger.info(f"Demonstrating prediction pipeline for {symbol}")
    
    # Prepare data
    X, y = prepare_data(symbol)
    
    if X is None or y is None:
        return
    
    # Create prediction pipeline
    pipeline = create_prediction_pipeline(
        use_ensemble=True,
        use_advanced_models=True,
        use_uncertainty=True,
        ensemble_type="weighted",
        uncertainty_method="bootstrap"
    )
    
    # Make predictions
    predictions = pipeline.predict(X)
    
    # Print results
    print("\nPrediction Pipeline Results:")
    print("=" * 50)
    print(f"Latest prediction: {predictions['prediction'].iloc[-1]:.4f}")
    
    if "lower_bound" in predictions.columns and "upper_bound" in predictions.columns:
        print(f"Confidence interval: [{predictions['lower_bound'].iloc[-1]:.4f}, {predictions['upper_bound'].iloc[-1]:.4f}]")
    
    if "uncertainty" in predictions.columns:
        print(f"Uncertainty: {predictions['uncertainty'].iloc[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y.values, 'b-', label='Actual')
    plt.plot(predictions["prediction"].values, 'r-', label='Prediction')
    
    if "lower_bound" in predictions.columns and "upper_bound" in predictions.columns:
        plt.fill_between(
            np.arange(len(predictions)),
            predictions["lower_bound"].values,
            predictions["upper_bound"].values,
            alpha=0.2, color='r',
            label='95% Confidence Interval'
        )
    
    plt.title(f'Prediction Pipeline for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/pipeline_{symbol}.png")
    
    print(f"\nPrediction pipeline results saved to results/plots/pipeline_{symbol}.png")


def main():
    """Main function."""
    # Create output directories
    os.makedirs("results/plots", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced models example")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data")
    parser.add_argument("--demo", type=str, default="all", help="Demo to run (finbert, tft, ensemble, uncertainty, pipeline, all)")
    
    args = parser.parse_args()
    
    # Run demos
    if args.demo == "finbert" or args.demo == "all":
        demo_finbert()
    
    if args.demo == "tft" or args.demo == "all":
        demo_tft(args.symbol)
    
    if args.demo == "ensemble" or args.demo == "all":
        demo_ensemble(args.symbol)
    
    if args.demo == "uncertainty" or args.demo == "all":
        demo_uncertainty(args.symbol)
    
    if args.demo == "pipeline" or args.demo == "all":
        demo_prediction_pipeline(args.symbol)


if __name__ == "__main__":
    main()
