"""
Script to tune hyperparameters for the prediction model.

This script performs hyperparameter tuning for the XGBoost model
using cross-validation and grid/random search.
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
from src.models.hyperparameter_tuning import (
    tune_xgboost, create_time_series_cv, get_financial_scorers,
    get_xgboost_param_grid, cross_validate_model, save_tuning_results
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(project_root) / "models"
TUNING_DIR = MODEL_DIR / "tuning"
N_FEATURES = 30  # Number of features to select
N_SPLITS = 5  # Number of cross-validation splits
N_ITER = 50  # Number of iterations for random search


def prepare_data_for_tuning(prediction_horizon=5):
    """
    Prepare data for hyperparameter tuning.
    
    Args:
        prediction_horizon: Number of days ahead to predict
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info("Preparing data for hyperparameter tuning")
    
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
    
    # Select features
    X_selected, selected_features = select_features(X, y, method='f_regression', k=N_FEATURES)
    
    logger.info(f"Prepared dataset with {len(X_selected)} samples and {len(selected_features)} features")
    
    return X_selected, y, selected_features


def tune_model_hyperparameters(X, y, feature_names, mode='random'):
    """
    Tune model hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        mode: 'grid' for grid search, 'random' for randomized search
        
    Returns:
        Tuple of (best model, tuning results)
    """
    logger.info(f"Tuning model hyperparameters using {mode} search")
    
    # Create cross-validation object
    cv = create_time_series_cv(n_splits=N_SPLITS)
    
    # Create scoring functions
    scorers = get_financial_scorers()
    
    # Create parameter grid
    param_grid = get_xgboost_param_grid(mode)
    
    # Tune XGBoost hyperparameters
    best_model, results = tune_xgboost(
        X, y,
        cv=cv,
        param_grid=param_grid,
        n_iter=N_ITER if mode == 'random' else None,
        scoring=scorers,
        n_jobs=-1,
        verbose=1,
        mode=mode
    )
    
    # Save tuning results
    os.makedirs(TUNING_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"xgboost_tuning_{mode}_{timestamp}.pkl"
    save_tuning_results(results, results_filename, str(TUNING_DIR))
    
    # Save best parameters
    best_params_path = TUNING_DIR / f"xgboost_best_params_{mode}_{timestamp}.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_model.get_params(), f, indent=2)
    
    logger.info(f"Best parameters saved to {best_params_path}")
    
    return best_model, results


def evaluate_tuned_model(model, X, y, feature_names):
    """
    Evaluate the tuned model using cross-validation.
    
    Args:
        model: Tuned model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        Cross-validation results
    """
    logger.info("Evaluating tuned model using cross-validation")
    
    # Create cross-validation object
    cv = create_time_series_cv(n_splits=N_SPLITS)
    
    # Create scoring functions
    scorers = get_financial_scorers()
    
    # Cross-validate model
    cv_results = cross_validate_model(
        model, X, y,
        cv=cv,
        scoring=scorers,
        n_jobs=-1
    )
    
    # Print results
    print("\nCross-Validation Results:")
    print("========================")
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            print(f"{metric}: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    
    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = TUNING_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Print top features
        print("\nTop 10 Features:")
        for i, (feature, importance) in enumerate(zip(importance_df['feature'].head(10), 
                                                   importance_df['importance'].head(10))):
            print(f"  {i+1}. {feature}: {importance:.6f}")
    
    return cv_results


def main():
    """Main function to tune and evaluate the model."""
    try:
        # Prepare data for tuning
        X, y, feature_names = prepare_data_for_tuning(PREDICTION_HORIZON)
        
        # Tune model hyperparameters
        best_model, tuning_results = tune_model_hyperparameters(X, y, feature_names, mode='random')
        
        # Evaluate tuned model
        cv_results = evaluate_tuned_model(best_model, X, y, feature_names)
        
        # Print summary
        print("\nHyperparameter Tuning Summary:")
        print("=============================")
        print(f"Best parameters: {best_model.get_params()}")
        
        # Get best score
        if isinstance(tuning_results.get('best_score'), float):
            print(f"Best score: {tuning_results['best_score']:.6f}")
        
        print("\nTuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
