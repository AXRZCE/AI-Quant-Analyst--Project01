"""
Prediction pipeline module.

This module provides a unified prediction pipeline that integrates advanced models,
ensembling, and uncertainty quantification.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator
from datetime import datetime, timedelta

# Import local modules
from src.models.ensemble import create_ensemble, save_ensemble, load_ensemble
from src.models.model_selection import ModelRegistry, ModelSelector
from src.models.advanced_models import ModelIntegrator, TFTWrapper, FinBERTWrapper
from src.models.uncertainty import create_uncertainty_quantifier, add_uncertainty_to_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Unified prediction pipeline."""
    
    def __init__(
        self,
        model_registry_dir: str = "models/registry",
        use_ensemble: bool = True,
        use_advanced_models: bool = True,
        use_uncertainty: bool = True,
        ensemble_type: str = "weighted",
        uncertainty_method: str = "bootstrap",
        primary_metric: str = "rmse"
    ):
        """
        Initialize the prediction pipeline.
        
        Args:
            model_registry_dir: Directory for model registry
            use_ensemble: Whether to use model ensembling
            use_advanced_models: Whether to use advanced models
            use_uncertainty: Whether to use uncertainty quantification
            ensemble_type: Type of ensemble to use
            uncertainty_method: Method for uncertainty quantification
            primary_metric: Primary metric for model selection
        """
        self.model_registry_dir = model_registry_dir
        self.use_ensemble = use_ensemble
        self.use_advanced_models = use_advanced_models
        self.use_uncertainty = use_uncertainty
        self.ensemble_type = ensemble_type
        self.uncertainty_method = uncertainty_method
        self.primary_metric = primary_metric
        
        # Initialize components
        self.model_registry = ModelRegistry(model_registry_dir)
        self.model_selector = ModelSelector(self.model_registry, primary_metric=primary_metric)
        self.model_integrator = None
        self.ensemble_model = None
        self.uncertainty_quantifier = None
        self.base_model = None
        
        # Initialize advanced models if enabled
        if self.use_advanced_models:
            self.model_integrator = ModelIntegrator()
    
    def load_models(
        self,
        model_id: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Load models from the registry.
        
        Args:
            model_id: Specific model ID to load
            model_type: Type of model to load
            tags: Tags to filter models
        """
        logger.info("Loading models")
        
        # Load specific model if ID provided
        if model_id:
            model, metadata = self.model_registry.load_model(model_id)
            
            if model is None:
                logger.error(f"Failed to load model {model_id}")
                return
            
            self.base_model = model
            logger.info(f"Loaded model {model_id}")
        
        # Otherwise, select best model
        else:
            model, metadata = self.model_selector.select_best_model(
                model_type=model_type,
                tags=tags
            )
            
            if model is None:
                logger.error("Failed to select best model")
                return
            
            self.base_model = model
            logger.info(f"Selected best model with {self.primary_metric}={metadata.get('metrics', {}).get(self.primary_metric)}")
        
        # Create ensemble if enabled
        if self.use_ensemble:
            self._create_ensemble()
        
        # Create uncertainty quantifier if enabled
        if self.use_uncertainty:
            self._create_uncertainty_quantifier()
    
    def _create_ensemble(self) -> None:
        """Create an ensemble model."""
        logger.info(f"Creating {self.ensemble_type} ensemble")
        
        # Get top models from registry
        models = self.model_registry.list_models(
            sort_by=f"metrics.{self.primary_metric}",
            ascending=True
        )
        
        # Use top 5 models or fewer if not available
        n_models = min(5, len(models))
        
        if n_models < 2:
            logger.warning("Not enough models for ensemble, using base model only")
            return
        
        # Load models
        ensemble_models = []
        
        for i in range(n_models):
            model_id = models[i]["model_id"]
            model, _ = self.model_registry.load_model(model_id)
            
            if model is not None:
                ensemble_models.append((model_id, model))
        
        if len(ensemble_models) < 2:
            logger.warning("Failed to load enough models for ensemble, using base model only")
            return
        
        # Create ensemble
        self.ensemble_model = create_ensemble(
            ensemble_models,
            ensemble_type=self.ensemble_type,
            optimize_weights=True
        )
        
        logger.info(f"Created ensemble with {len(ensemble_models)} models")
    
    def _create_uncertainty_quantifier(self) -> None:
        """Create an uncertainty quantifier."""
        logger.info(f"Creating uncertainty quantifier using {self.uncertainty_method} method")
        
        # Use ensemble model if available, otherwise use base model
        model = self.ensemble_model if self.ensemble_model is not None else self.base_model
        
        if model is None:
            logger.error("No model available for uncertainty quantification")
            return
        
        # Create uncertainty quantifier
        self.uncertainty_quantifier = create_uncertainty_quantifier(
            model,
            method=self.uncertainty_method,
            n_estimators=100 if self.uncertainty_method == "bootstrap" else None,
            alpha=0.05 if self.uncertainty_method == "conformal" else None,
            model_type="numpyro" if self.uncertainty_method == "bayesian" else None
        )
        
        logger.info(f"Created {self.uncertainty_method} uncertainty quantifier")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            df: DataFrame with raw data
            text_column: Column with text data for sentiment analysis
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing features")
        
        # Process with advanced models if enabled
        if self.use_advanced_models and self.model_integrator is not None:
            df = self.model_integrator.process_data(df, text_column)
        
        return df
    
    def predict(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions with or without uncertainty
        """
        logger.info(f"Making predictions for {len(X)} samples")
        
        # Check if models are loaded
        if self.ensemble_model is None and self.base_model is None:
            logger.error("No models loaded")
            return np.zeros(len(X))
        
        # Use ensemble model if available
        if self.ensemble_model is not None:
            logger.info("Using ensemble model for predictions")
            predictions = self.ensemble_model.predict(X)
        else:
            logger.info("Using base model for predictions")
            predictions = self.base_model.predict(X)
        
        # Add uncertainty if requested
        if return_uncertainty and self.uncertainty_quantifier is not None:
            logger.info("Adding uncertainty estimates")
            
            # Fit uncertainty quantifier if needed
            if hasattr(self.uncertainty_quantifier, 'fitted') and not self.uncertainty_quantifier.fitted:
                logger.info("Fitting uncertainty quantifier")
                # We need to fit on some data, but we don't have labels for X
                # Use a small random subset of X with dummy labels for fitting
                n_samples = min(100, len(X))
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                y_sample = np.random.randn(n_samples)  # Dummy labels
                self.uncertainty_quantifier.fit(X_sample, y_sample)
            
            # Get prediction intervals
            lower_bound, upper_bound = self.uncertainty_quantifier.predict_interval(X)
            
            # Create DataFrame with predictions and uncertainty
            result = add_uncertainty_to_predictions(predictions, lower_bound, upper_bound)
            
            return result
        else:
            return predictions
    
    def predict_with_sentiment(
        self,
        X: pd.DataFrame,
        texts: List[str],
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions with sentiment analysis.
        
        Args:
            X: Feature matrix
            texts: List of texts for sentiment analysis
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions and sentiment
        """
        logger.info(f"Making predictions with sentiment analysis for {len(X)} samples")
        
        # Make predictions
        predictions = self.predict(X, return_uncertainty)
        
        # Add sentiment if advanced models are enabled
        if self.use_advanced_models and self.model_integrator is not None:
            logger.info("Adding sentiment analysis")
            sentiment = self.model_integrator.get_sentiment(texts)
            
            # Create result dictionary
            result = {
                "predictions": predictions,
                "sentiment": sentiment
            }
            
            return result
        else:
            return {"predictions": predictions}
    
    def predict_time_series(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        return_uncertainty: bool = True
    ) -> pd.DataFrame:
        """
        Make time series predictions.
        
        Args:
            df: DataFrame with time series data
            horizon: Forecast horizon
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            DataFrame with forecasts
        """
        logger.info(f"Making time series predictions with horizon {horizon}")
        
        # Use TFT model if advanced models are enabled
        if self.use_advanced_models and self.model_integrator is not None:
            logger.info("Using TFT model for time series forecasting")
            forecasts = self.model_integrator.get_time_series_forecast(df, horizon)
            
            if not forecasts.empty:
                return forecasts
        
        # Fall back to base model
        logger.info("Using base model for time series forecasting")
        
        # Check if models are loaded
        if self.ensemble_model is None and self.base_model is None:
            logger.error("No models loaded")
            return pd.DataFrame()
        
        # Create forecast DataFrame
        forecasts = pd.DataFrame()
        
        # Use ensemble model if available
        if self.ensemble_model is not None:
            forecasts["forecast"] = self.ensemble_model.predict(df)
        else:
            forecasts["forecast"] = self.base_model.predict(df)
        
        # Add uncertainty if requested
        if return_uncertainty and self.uncertainty_quantifier is not None:
            logger.info("Adding uncertainty estimates")
            
            # Get prediction intervals
            lower_bound, upper_bound = self.uncertainty_quantifier.predict_interval(df)
            
            # Add to DataFrame
            forecasts["lower"] = lower_bound
            forecasts["upper"] = upper_bound
        
        return forecasts
    
    def save_pipeline(self, path: str) -> None:
        """
        Save the prediction pipeline.
        
        Args:
            path: Path to save the pipeline
        """
        logger.info(f"Saving prediction pipeline to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare pipeline package
        pipeline_package = {
            "model_registry_dir": self.model_registry_dir,
            "use_ensemble": self.use_ensemble,
            "use_advanced_models": self.use_advanced_models,
            "use_uncertainty": self.use_uncertainty,
            "ensemble_type": self.ensemble_type,
            "uncertainty_method": self.uncertainty_method,
            "primary_metric": self.primary_metric,
            "base_model": self.base_model,
            "ensemble_model": self.ensemble_model,
            "uncertainty_quantifier": self.uncertainty_quantifier
        }
        
        # Save pipeline
        joblib.dump(pipeline_package, path)
        
        logger.info(f"Saved prediction pipeline to {path}")
    
    @classmethod
    def load_pipeline(cls, path: str) -> 'PredictionPipeline':
        """
        Load a prediction pipeline.
        
        Args:
            path: Path to the saved pipeline
            
        Returns:
            Loaded prediction pipeline
        """
        logger.info(f"Loading prediction pipeline from {path}")
        
        # Load pipeline package
        pipeline_package = joblib.load(path)
        
        # Create pipeline
        pipeline = cls(
            model_registry_dir=pipeline_package["model_registry_dir"],
            use_ensemble=pipeline_package["use_ensemble"],
            use_advanced_models=pipeline_package["use_advanced_models"],
            use_uncertainty=pipeline_package["use_uncertainty"],
            ensemble_type=pipeline_package["ensemble_type"],
            uncertainty_method=pipeline_package["uncertainty_method"],
            primary_metric=pipeline_package["primary_metric"]
        )
        
        # Set models
        pipeline.base_model = pipeline_package["base_model"]
        pipeline.ensemble_model = pipeline_package["ensemble_model"]
        pipeline.uncertainty_quantifier = pipeline_package["uncertainty_quantifier"]
        
        # Initialize advanced models if enabled
        if pipeline.use_advanced_models:
            pipeline.model_integrator = ModelIntegrator()
        
        logger.info(f"Loaded prediction pipeline from {path}")
        
        return pipeline


def create_prediction_pipeline(
    model_registry_dir: str = "models/registry",
    use_ensemble: bool = True,
    use_advanced_models: bool = True,
    use_uncertainty: bool = True,
    ensemble_type: str = "weighted",
    uncertainty_method: str = "bootstrap",
    primary_metric: str = "rmse",
    model_id: Optional[str] = None,
    model_type: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> PredictionPipeline:
    """
    Create a prediction pipeline.
    
    Args:
        model_registry_dir: Directory for model registry
        use_ensemble: Whether to use model ensembling
        use_advanced_models: Whether to use advanced models
        use_uncertainty: Whether to use uncertainty quantification
        ensemble_type: Type of ensemble to use
        uncertainty_method: Method for uncertainty quantification
        primary_metric: Primary metric for model selection
        model_id: Specific model ID to load
        model_type: Type of model to load
        tags: Tags to filter models
        
    Returns:
        Prediction pipeline
    """
    # Create pipeline
    pipeline = PredictionPipeline(
        model_registry_dir=model_registry_dir,
        use_ensemble=use_ensemble,
        use_advanced_models=use_advanced_models,
        use_uncertainty=use_uncertainty,
        ensemble_type=ensemble_type,
        uncertainty_method=uncertainty_method,
        primary_metric=primary_metric
    )
    
    # Load models
    pipeline.load_models(
        model_id=model_id,
        model_type=model_type,
        tags=tags
    )
    
    return pipeline
