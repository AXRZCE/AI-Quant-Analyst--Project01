"""
Script to train the prediction model with MLflow tracking.

This script trains the XGBoost model and tracks the experiment with MLflow,
including parameters, metrics, artifacts, and the model itself.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import project modules
from src.models.train_model import fetch_training_data, fetch_sentiment_data, SYMBOLS, LOOKBACK_DAYS, PREDICTION_HORIZON
from src.models.feature_engineering import get_feature_target_split, select_features
from src.models.model_evaluation import evaluate_regression_model, generate_evaluation_report
from src.models.mlflow_utils import (
    setup_mlflow, start_run, log_params, log_metrics, log_model,
    log_feature_importance, log_predictions, log_dataset_info,
    track_experiment
)

# Import XGBoost
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(project_root) / "models"
MLFLOW_DIR = MODEL_DIR / "mlflow"
EXPERIMENT_NAME = "financial_forecasting"
N_FEATURES = 30  # Number of features to select


def prepare_data_for_training(prediction_horizon=5):
    """
    Prepare data for model training.
    
    Args:
        prediction_horizon: Number of days ahead to predict
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, prices)
    """
    logger.info("Preparing data for model training")
    
    # Fetch training data with enhanced features
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
        train_size = int(0.8 * len(sorted_indices))
        train_indices = sorted_indices[:train_size]
        test_indices = sorted_indices[train_size:]
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        
        # Get prices for train and test sets
        if prices is not None:
            prices_test = prices.loc[test_indices]
        else:
            prices_test = None
            
        logger.info(f"Using time-based split: {len(X_train)} training samples, {len(X_test)} testing samples")
    else:
        # Fall back to random split if timestamp not available
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get prices for test set (not ideal with random split)
        prices_test = None
        
        logger.info(f"Using random split: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    # Select features
    X_train_selected, selected_features = select_features(X_train, y_train, method='f_regression', k=N_FEATURES)
    X_test_selected = X_test[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features")
    
    return X_train_selected, X_test_selected, y_train, y_test, selected_features, prices_test


def train_model_with_mlflow():
    """
    Train the model and track the experiment with MLflow.
    
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    logger.info("Training model with MLflow tracking")
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names, prices_test = prepare_data_for_training(PREDICTION_HORIZON)
    
    # Set up MLflow
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow_uri = f"file://{MLFLOW_DIR.absolute()}"
    setup_mlflow(mlflow_uri, EXPERIMENT_NAME)
    
    # Define model parameters
    params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"xgboost_baseline_{timestamp}"
    
    # Start MLflow run
    with start_run(run_name=run_name, experiment_name=EXPERIMENT_NAME, tracking_uri=mlflow_uri) as run:
        # Log dataset info
        log_dataset_info(X_train, y_train, X_test, y_test)
        
        # Log parameters
        log_params(params)
        log_params({
            'prediction_horizon': PREDICTION_HORIZON,
            'symbols': SYMBOLS,
            'lookback_days': LOOKBACK_DAYS,
            'n_features': N_FEATURES,
            'feature_selection_method': 'f_regression'
        })
        
        # Create and train model
        model = XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Get best iteration
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        
        # Evaluate model
        metrics = evaluate_regression_model(model, X_test, y_test, prices_test)
        
        # Log metrics
        log_metrics(metrics)
        
        # Log feature importance
        log_feature_importance(feature_names, model.feature_importances_)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log predictions
        log_predictions(y_test, y_pred)
        
        # Log model
        log_model(
            model,
            model_name="xgboost_model",
            X=X_test,
            y=y_test,
            registered_model_name="financial_forecasting_model"
        )
        
        # Print summary
        print("\nModel Training Summary (with MLflow tracking):")
        print("=============================================")
        print(f"Trained on {len(X_train)} samples with {len(feature_names)} selected features")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"\nTop 5 Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(5), 
                                                  feature_importance['importance'].head(5))):
            print(f"  {i+1}. {feature}: {importance:.6f}")
        
        print(f"\nMLflow tracking URI: {mlflow_uri}")
        print(f"Run ID: {run.info.run_id}")
        
        return model, metrics


def main():
    """Main function to train the model with MLflow tracking."""
    try:
        # Train model with MLflow tracking
        model, metrics = train_model_with_mlflow()
        
        # Save model and metrics
        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = MODEL_DIR / f"xgboost_model_{timestamp}.pkl"
        import joblib
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = MODEL_DIR / f"xgboost_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("Model training with MLflow tracking completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training with MLflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
