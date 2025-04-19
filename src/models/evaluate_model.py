"""
Script to evaluate a trained prediction model.

This script loads a trained model and evaluates it on test data,
generating comprehensive evaluation metrics and visualizations.
"""

import os
import sys
import logging
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import project modules
from src.models.train_model import fetch_training_data, fetch_sentiment_data, SYMBOLS, LOOKBACK_DAYS, PREDICTION_HORIZON
from src.models.feature_engineering import get_feature_target_split
from src.models.model_evaluation import (
    evaluate_regression_model, generate_evaluation_report,
    plot_predictions, plot_feature_importance, plot_cumulative_returns
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(project_root) / "models"
REPORTS_DIR = MODEL_DIR / "reports"


def load_model_and_features(model_path):
    """
    Load a trained model and its selected features.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, feature_names)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Try to load feature names
    features_path = model_path.with_suffix('').with_name(f"{model_path.stem}_features.json")
    
    if features_path.exists():
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Loaded {len(feature_names)} feature names from {features_path}")
    else:
        logger.warning(f"Feature names file not found at {features_path}")
        feature_names = None
    
    return model, feature_names


def prepare_test_data(feature_names=None, prediction_horizon=5, test_size=0.2):
    """
    Prepare test data for model evaluation.
    
    Args:
        feature_names: List of feature names to use (if None, use all features)
        prediction_horizon: Number of days ahead to predict
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_test, y_test, prices_test)
    """
    logger.info("Preparing test data for model evaluation")
    
    # Fetch data
    data = fetch_training_data(SYMBOLS, LOOKBACK_DAYS, prediction_horizon)
    
    # Fetch sentiment data
    sentiment = fetch_sentiment_data(SYMBOLS, LOOKBACK_DAYS)
    
    # Merge sentiment data if available
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
    
    # Get price series for financial metrics
    prices = data['close'] if 'close' in data.columns else None
    
    # Split data into training and testing sets using time-based split
    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        sorted_indices = data.sort_values('timestamp').index
        train_size = int((1 - test_size) * len(sorted_indices))
        test_indices = sorted_indices[train_size:]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]
        
        # Get prices for test set
        if prices is not None:
            prices_test = prices.loc[test_indices]
        else:
            prices_test = None
            
        logger.info(f"Using time-based split: {len(X_test)} testing samples")
    else:
        # Fall back to random split if timestamp not available
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Get prices for test set (not ideal with random split)
        prices_test = None
        
        logger.info(f"Using random split: {len(X_test)} testing samples")
    
    # Filter features if feature names are provided
    if feature_names is not None:
        # Check if all feature names exist in X_test
        missing_features = [f for f in feature_names if f not in X_test.columns]
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
            # Use only available features
            feature_names = [f for f in feature_names if f in X_test.columns]
        
        X_test = X_test[feature_names]
        logger.info(f"Filtered test data to {len(feature_names)} features")
    
    return X_test, y_test, prices_test


def evaluate_model(model, X_test, y_test, prices_test=None, feature_names=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test target vector
        prices_test: Test price series (optional)
        feature_names: List of feature names (optional)
        
    Returns:
        Evaluation report
    """
    logger.info("Evaluating model on test data")
    
    # Generate evaluation report
    report = generate_evaluation_report(
        model, X_test, y_test,
        feature_names=feature_names or X_test.columns.tolist(),
        prices=prices_test
    )
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print("========================")
    for name, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.6f}")
    
    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_path = REPORTS_DIR / f"evaluation_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                  for k, v in report["metrics"].items()}, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Save plots
    for name, fig in report["plots"].items():
        plot_path = REPORTS_DIR / f"{name}_{timestamp}.png"
        fig.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
    
    return report


def main():
    """Main function to evaluate a trained model."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate a trained prediction model")
        parser.add_argument("--model", type=str, required=True, help="Path to the model file")
        parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON, help="Prediction horizon (days)")
        parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
        args = parser.parse_args()
        
        # Load model and features
        model_path = Path(args.model)
        model, feature_names = load_model_and_features(model_path)
        
        # Prepare test data
        X_test, y_test, prices_test = prepare_test_data(
            feature_names=feature_names,
            prediction_horizon=args.horizon,
            test_size=args.test_size
        )
        
        # Evaluate model
        report = evaluate_model(model, X_test, y_test, prices_test, feature_names)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
