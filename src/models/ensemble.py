"""
Model ensembling module.

This module provides functionality for creating and using model ensembles,
including voting, stacking, and blending ensembles.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Weighted ensemble of regression models."""
    
    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
        optimize_weights: bool = False
    ):
        """
        Initialize the weighted ensemble.
        
        Args:
            models: List of (name, model) tuples
            weights: List of weights for each model (if None, use equal weights)
            optimize_weights: Whether to optimize weights during fitting
        """
        self.models = models
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.fitted_weights_ = None
        self.model_names_ = [name for name, _ in models]
        self.base_models_ = [model for _, model in models]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedEnsemble':
        """
        Fit the ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted ensemble
        """
        logger.info(f"Fitting weighted ensemble with {len(self.models)} models")
        
        # Fit each base model
        for name, model in self.models:
            logger.info(f"Fitting {name}")
            model.fit(X, y)
        
        # Optimize weights if requested
        if self.optimize_weights:
            logger.info("Optimizing ensemble weights")
            self.fitted_weights_ = self._optimize_weights(X, y)
        else:
            # Use provided weights or equal weights
            if self.weights is None:
                self.fitted_weights_ = np.ones(len(self.models)) / len(self.models)
            else:
                # Normalize weights to sum to 1
                self.fitted_weights_ = np.array(self.weights) / np.sum(self.weights)
        
        logger.info(f"Ensemble weights: {dict(zip(self.model_names_, self.fitted_weights_))}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        # Get predictions from each model
        predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        
        # Apply weights
        weighted_predictions = predictions * self.fitted_weights_
        
        # Sum weighted predictions
        return np.sum(weighted_predictions, axis=1)
    
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        # Get predictions from each model
        predictions = np.column_stack([
            model.predict_interval(X, alpha=alpha)[0] if hasattr(model, 'predict_interval') 
            else model.predict(X) 
            for model in self.base_models_
        ])
        
        # Get upper bounds from each model
        upper_bounds = np.column_stack([
            model.predict_interval(X, alpha=alpha)[1] if hasattr(model, 'predict_interval') 
            else model.predict(X) 
            for model in self.base_models_
        ])
        
        # Calculate weighted predictions and intervals
        weighted_predictions = predictions * self.fitted_weights_
        weighted_upper = upper_bounds * self.fitted_weights_
        
        # Sum weighted predictions
        mean_pred = np.sum(weighted_predictions, axis=1)
        
        # Calculate uncertainty as weighted average of model uncertainties
        uncertainty = np.sum(weighted_upper, axis=1) - mean_pred
        
        # Return lower and upper bounds
        return mean_pred - uncertainty, mean_pred + uncertainty
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Optimize ensemble weights.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Optimized weights
        """
        # Get predictions from each model
        predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        
        # Use linear regression to find optimal weights
        lr = LinearRegression(fit_intercept=False, positive=True)
        lr.fit(predictions, y)
        
        # Get weights
        weights = lr.coef_
        
        # Normalize weights to sum to 1
        return weights / np.sum(weights)
    
    def get_model_importances(self) -> Dict[str, float]:
        """
        Get model importances.
        
        Returns:
            Dictionary of model names and importances
        """
        return dict(zip(self.model_names_, self.fitted_weights_))


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """Stacked ensemble of regression models."""
    
    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        meta_model: Optional[BaseEstimator] = None,
        cv: int = 5,
        use_features: bool = True
    ):
        """
        Initialize the stacked ensemble.
        
        Args:
            models: List of (name, model) tuples
            meta_model: Meta-model for stacking (if None, use LinearRegression)
            cv: Number of cross-validation folds
            use_features: Whether to use original features in meta-model
        """
        self.models = models
        self.meta_model = meta_model or LinearRegression()
        self.cv = cv
        self.use_features = use_features
        self.model_names_ = [name for name, _ in models]
        self.base_models_ = [model for _, model in models]
        self.stacking_regressor_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackedEnsemble':
        """
        Fit the ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted ensemble
        """
        logger.info(f"Fitting stacked ensemble with {len(self.models)} models")
        
        # Create cross-validation object
        if isinstance(X, pd.DataFrame) and 'timestamp' in X.columns:
            # Use time series split for time series data
            cv = TimeSeriesSplit(n_splits=self.cv)
        else:
            # Use regular K-fold for other data
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Create stacking regressor
        self.stacking_regressor_ = StackingRegressor(
            estimators=self.models,
            final_estimator=self.meta_model,
            cv=cv,
            passthrough=self.use_features
        )
        
        # Fit stacking regressor
        self.stacking_regressor_.fit(X, y)
        
        logger.info("Stacked ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        return self.stacking_regressor_.predict(X)
    
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        # Get predictions
        predictions = self.predict(X)
        
        # Get predictions from each base model
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        
        # Calculate standard deviation of base predictions
        std_dev = np.std(base_predictions, axis=1)
        
        # Calculate z-score for the given alpha
        z_score = 1.96  # Approximately 95% confidence interval
        
        # Calculate lower and upper bounds
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        return lower_bound, upper_bound
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances from the meta-model.
        
        Returns:
            Dictionary of feature names and importances
        """
        if not hasattr(self.meta_model, 'coef_'):
            logger.warning("Meta-model does not have feature importances")
            return {}
        
        # Get feature importances
        importances = self.meta_model.coef_
        
        # Get feature names
        if self.use_features:
            # Feature names include base model predictions and original features
            feature_names = self.model_names_ + [f"feature_{i}" for i in range(importances.shape[0] - len(self.model_names_))]
        else:
            # Feature names are just base model predictions
            feature_names = self.model_names_
        
        return dict(zip(feature_names, importances))


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """Blending ensemble of regression models."""
    
    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        meta_model: Optional[BaseEstimator] = None,
        val_size: float = 0.2
    ):
        """
        Initialize the blending ensemble.
        
        Args:
            models: List of (name, model) tuples
            meta_model: Meta-model for blending (if None, use LinearRegression)
            val_size: Validation set size for blending
        """
        self.models = models
        self.meta_model = meta_model or LinearRegression()
        self.val_size = val_size
        self.model_names_ = [name for name, _ in models]
        self.base_models_ = [model for _, model in models]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit the ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted ensemble
        """
        logger.info(f"Fitting blending ensemble with {len(self.models)} models")
        
        # Split data into training and validation sets
        if isinstance(X, pd.DataFrame) and 'timestamp' in X.columns:
            # Use time-based split for time series data
            split_idx = int((1 - self.val_size) * len(X))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Use random split for other data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, random_state=42
            )
        
        # Fit base models on training data
        for name, model in self.models:
            logger.info(f"Fitting {name}")
            model.fit(X_train, y_train)
        
        # Get predictions on validation data
        val_predictions = np.column_stack([model.predict(X_val) for _, model in self.models])
        
        # Fit meta-model on validation predictions
        logger.info("Fitting meta-model")
        self.meta_model.fit(val_predictions, y_val)
        
        # Refit base models on all data
        for name, model in self.models:
            logger.info(f"Refitting {name} on all data")
            model.fit(X, y)
        
        logger.info("Blending ensemble fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        # Get predictions from base models
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        
        # Make meta-model predictions
        return self.meta_model.predict(base_predictions)
    
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        # Get predictions
        predictions = self.predict(X)
        
        # Get predictions from each base model
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        
        # Calculate standard deviation of base predictions
        std_dev = np.std(base_predictions, axis=1)
        
        # Calculate z-score for the given alpha
        z_score = 1.96  # Approximately 95% confidence interval
        
        # Calculate lower and upper bounds
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        return lower_bound, upper_bound
    
    def get_model_importances(self) -> Dict[str, float]:
        """
        Get model importances from the meta-model.
        
        Returns:
            Dictionary of model names and importances
        """
        if not hasattr(self.meta_model, 'coef_'):
            logger.warning("Meta-model does not have feature importances")
            return {}
        
        # Get model importances
        importances = self.meta_model.coef_
        
        return dict(zip(self.model_names_, importances))


def create_ensemble(
    models: List[Tuple[str, BaseEstimator]],
    ensemble_type: str = 'weighted',
    **kwargs
) -> BaseEstimator:
    """
    Create an ensemble model.
    
    Args:
        models: List of (name, model) tuples
        ensemble_type: Type of ensemble ('weighted', 'stacked', or 'blending')
        **kwargs: Additional arguments for the ensemble
        
    Returns:
        Ensemble model
    """
    logger.info(f"Creating {ensemble_type} ensemble with {len(models)} models")
    
    if ensemble_type == 'weighted':
        return WeightedEnsemble(models, **kwargs)
    elif ensemble_type == 'stacked':
        return StackedEnsemble(models, **kwargs)
    elif ensemble_type == 'blending':
        return BlendingEnsemble(models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def save_ensemble(
    ensemble: BaseEstimator,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save the ensemble model.
    
    Args:
        ensemble: Ensemble model
        output_path: Path to save the model
        metadata: Additional metadata to save with the model
    """
    logger.info(f"Saving ensemble model to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare model package
    model_package = {
        "model": ensemble,
        "metadata": metadata or {}
    }
    
    # Save model
    joblib.dump(model_package, output_path)
    
    # Save model info
    info_path = output_path.replace('.pkl', '_info.json')
    
    # Extract model info
    model_info = {
        "ensemble_type": ensemble.__class__.__name__,
        "base_models": getattr(ensemble, 'model_names_', []),
        "metadata": metadata or {}
    }
    
    # Add model importances if available
    if hasattr(ensemble, 'get_model_importances'):
        try:
            model_info["model_importances"] = ensemble.get_model_importances()
        except:
            pass
    
    # Save model info
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Ensemble model info saved to {info_path}")


def load_ensemble(model_path: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Load an ensemble model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (ensemble model, metadata)
    """
    logger.info(f"Loading ensemble model from {model_path}")
    
    # Load model package
    model_package = joblib.load(model_path)
    
    # Extract model and metadata
    if isinstance(model_package, dict) and "model" in model_package:
        ensemble = model_package["model"]
        metadata = model_package.get("metadata", {})
    else:
        # If not a package, assume it's the model directly
        ensemble = model_package
        metadata = {}
    
    return ensemble, metadata


def evaluate_ensemble(
    ensemble: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Evaluate an ensemble model.
    
    Args:
        ensemble: Ensemble model
        X: Feature matrix
        y: Target vector
        metrics: Dictionary of metric names and functions
        
    Returns:
        Dictionary of metric names and values
    """
    logger.info("Evaluating ensemble model")
    
    # Use default metrics if not provided
    if metrics is None:
        metrics = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score
        }
    
    # Make predictions
    y_pred = ensemble.predict(X)
    
    # Calculate metrics
    results = {}
    for name, metric_fn in metrics.items():
        results[name] = metric_fn(y, y_pred)
    
    # Add RMSE
    if 'mse' in results:
        results['rmse'] = np.sqrt(results['mse'])
    
    # Log results
    for name, value in results.items():
        logger.info(f"{name}: {value:.6f}")
    
    return results
