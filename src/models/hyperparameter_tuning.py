"""
Hyperparameter tuning module for financial time series models.

This module provides functions for tuning hyperparameters of machine learning models
for financial time series prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
import time
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.base import BaseEstimator

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

# Configure logging
logger = logging.getLogger(__name__)


def create_time_series_cv(n_splits: int = 5, test_size: Optional[int] = None) -> TimeSeriesSplit:
    """
    Create a time series cross-validation object.
    
    Args:
        n_splits: Number of splits
        test_size: Size of the test set in each split
        
    Returns:
        TimeSeriesSplit object
    """
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)


def directional_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Directional accuracy score
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return np.mean(true_direction == pred_direction)


def get_financial_scorers() -> Dict[str, Any]:
    """
    Get scoring functions for financial time series.
    
    Returns:
        Dictionary of scorer names and functions
    """
    scorers = {
        'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
        'r2': make_scorer(r2_score),
        'directional_accuracy': make_scorer(directional_accuracy_score)
    }
    
    return scorers


def get_xgboost_param_grid(mode: str = 'grid') -> Dict[str, Any]:
    """
    Get parameter grid for XGBoost.
    
    Args:
        mode: 'grid' for grid search, 'random' for randomized search
        
    Returns:
        Parameter grid
    """
    if mode == 'grid':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
    else:  # random search
        param_grid = {
            'n_estimators': np.arange(50, 500, 50),
            'max_depth': np.arange(3, 10),
            'learning_rate': np.logspace(-3, -1, 10),
            'subsample': np.linspace(0.6, 1.0, 5),
            'colsample_bytree': np.linspace(0.6, 1.0, 5),
            'min_child_weight': np.arange(1, 10),
            'gamma': np.linspace(0, 0.5, 10)
        }
    
    return param_grid


def tune_xgboost(X: pd.DataFrame, y: pd.Series, 
               cv: Optional[Any] = None, 
               param_grid: Optional[Dict[str, Any]] = None,
               n_iter: int = 20,
               scoring: Union[str, List[str], Dict[str, Any]] = 'neg_mean_squared_error',
               n_jobs: int = -1,
               verbose: int = 1,
               mode: str = 'random',
               is_classification: bool = False) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Tune XGBoost hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv: Cross-validation object
        param_grid: Parameter grid
        n_iter: Number of iterations for randomized search
        scoring: Scoring function(s)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        mode: 'grid' for grid search, 'random' for randomized search
        is_classification: Whether to use XGBClassifier instead of XGBRegressor
        
    Returns:
        Tuple of (best model, results)
    """
    logger.info(f"Tuning XGBoost hyperparameters using {mode} search")
    
    # Create cross-validation object if not provided
    if cv is None:
        cv = create_time_series_cv()
    
    # Create parameter grid if not provided
    if param_grid is None:
        param_grid = get_xgboost_param_grid(mode)
    
    # Create model
    if is_classification:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = XGBRegressor()
    
    # Create search object
    if mode == 'grid':
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
            refit=True if isinstance(scoring, str) else False
        )
    else:  # random search
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
            refit=True if isinstance(scoring, str) else False
        )
    
    # Fit search
    start_time = time.time()
    search.fit(X, y)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
    
    # Get best parameters
    logger.info(f"Best parameters: {search.best_params_}")
    
    # Get best score
    if isinstance(scoring, str):
        logger.info(f"Best score ({scoring}): {search.best_score_:.6f}")
    
    # Create best model
    if isinstance(scoring, str):
        best_model = search.best_estimator_
    else:
        # If multiple scoring metrics were used, refit with best parameters
        if is_classification:
            best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **search.best_params_)
        else:
            best_model = XGBRegressor(**search.best_params_)
        
        best_model.fit(X, y)
    
    # Prepare results
    results = {
        'best_params': search.best_params_,
        'cv_results': search.cv_results_,
        'elapsed_time': elapsed_time
    }
    
    if isinstance(scoring, str):
        results['best_score'] = search.best_score_
    
    return best_model, results


def tune_model_with_early_stopping(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 param_grid: Dict[str, Any],
                                 is_classification: bool = False,
                                 early_stopping_rounds: int = 10,
                                 verbose: bool = True) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Tune model with early stopping.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_val: Validation feature matrix
        y_val: Validation target vector
        param_grid: Parameter grid
        is_classification: Whether to use XGBClassifier instead of XGBRegressor
        early_stopping_rounds: Number of rounds for early stopping
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best model, results)
    """
    logger.info("Tuning model with early stopping")
    
    # Create model
    if is_classification:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = XGBRegressor()
    
    # Set parameters
    model.set_params(**param_grid)
    
    # Create evaluation set
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    # Fit model with early stopping
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='rmse' if not is_classification else 'logloss',
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best iteration: {model.best_iteration}")
    
    # Prepare results
    results = {
        'params': param_grid,
        'best_iteration': model.best_iteration,
        'elapsed_time': elapsed_time
    }
    
    if hasattr(model, 'evals_result'):
        results['evals_result'] = model.evals_result()
    
    return model, results


def cross_validate_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                       cv: Optional[Any] = None,
                       scoring: Union[str, List[str], Dict[str, Any]] = None,
                       n_jobs: int = -1,
                       verbose: int = 0) -> Dict[str, np.ndarray]:
    """
    Cross-validate a model.
    
    Args:
        model: Model to cross-validate
        X: Feature matrix
        y: Target vector
        cv: Cross-validation object
        scoring: Scoring function(s)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info("Cross-validating model")
    
    # Create cross-validation object if not provided
    if cv is None:
        cv = create_time_series_cv()
    
    # Create scoring functions if not provided
    if scoring is None:
        scoring = get_financial_scorers()
    
    # Cross-validate model
    cv_results = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Log results
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            logger.info(f"{metric}: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    
    return cv_results


def save_tuning_results(results: Dict[str, Any], filename: str, directory: str = 'models/tuning') -> str:
    """
    Save tuning results to disk.
    
    Args:
        results: Tuning results
        filename: Filename
        directory: Directory to save to
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create full path
    path = Path(directory) / filename
    
    # Save results
    joblib.dump(results, path)
    logger.info(f"Tuning results saved to {path}")
    
    return str(path)


def load_tuning_results(filename: str, directory: str = 'models/tuning') -> Dict[str, Any]:
    """
    Load tuning results from disk.
    
    Args:
        filename: Filename
        directory: Directory to load from
        
    Returns:
        Tuning results
    """
    # Create full path
    path = Path(directory) / filename
    
    # Load results
    results = joblib.load(path)
    logger.info(f"Tuning results loaded from {path}")
    
    return results


def get_best_params_from_results(results: Dict[str, Any], scoring: str = 'test_neg_rmse') -> Dict[str, Any]:
    """
    Get best parameters from tuning results.
    
    Args:
        results: Tuning results
        scoring: Scoring metric to use
        
    Returns:
        Best parameters
    """
    # Get CV results
    cv_results = results['cv_results']
    
    # Find index of best score
    best_index = np.argmax(cv_results[f'mean_{scoring}'])
    
    # Get best parameters
    best_params = {}
    for param in cv_results['params'][best_index]:
        best_params[param] = cv_results['params'][best_index][param]
    
    return best_params


def plot_tuning_results(results: Dict[str, Any], param_name: str, scoring: str = 'mean_test_score') -> Any:
    """
    Plot tuning results for a specific parameter.
    
    Args:
        results: Tuning results
        param_name: Parameter name to plot
        scoring: Scoring metric to use
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Get CV results
    cv_results = results['cv_results']
    
    # Get parameter values
    param_values = [params[param_name] for params in cv_results['params']]
    
    # Get scores
    scores = cv_results[scoring]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scores
    ax.plot(param_values, scores, 'o-')
    
    # Add labels
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(f"Effect of {param_name} on {scoring}")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig
