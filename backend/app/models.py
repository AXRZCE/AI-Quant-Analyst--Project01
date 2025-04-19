"""Model schemas and prediction logic for the API.

This module provides Pydantic models for request/response validation,
and functionality for loading and using machine learning models.
"""

import os
import logging
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from pydantic import BaseModel, Field, validator

from .config import settings
from .cache import cached

# Configure logging
logger = logging.getLogger(__name__)


# Request/Response Models
class PredictionRequest(BaseModel):
    """Model for prediction request."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    days: int = Field(1, description="Number of days of historical data to use", ge=1, le=30)
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features for prediction")

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        return v.upper()


class PredictionResponse(BaseModel):
    """Model for prediction response."""
    timestamps: List[str] = Field(..., description="Timestamps for the historical data")
    prices: List[float] = Field(..., description="Historical closing prices")
    ma_5: List[float] = Field(..., description="5-day moving average")
    rsi_14: List[float] = Field(..., description="14-day relative strength index")
    sentiment: Dict[str, float] = Field(..., description="Sentiment analysis results")
    prediction: float = Field(..., description="Predicted return")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval for prediction")
    feature_importance: Optional[List[Dict[str, Any]]] = Field(None, description="Feature importance")


class ModelInfoResponse(BaseModel):
    """Model for model info response."""
    model_path: str = Field(..., description="Path to the model file")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_type: Optional[str] = Field(None, description="Type of the model")
    features: Optional[List[str]] = Field(None, description="Features used by the model")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Model metrics")
    fallback_enabled: bool = Field(..., description="Whether fallback model is enabled")


# Model Manager
class ModelManager:
    """Model manager for loading and using machine learning models."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model manager.

        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path or settings.model.MODEL_PATH
        self.model = None
        self.features = None
        self.metrics = None
        self.fallback_enabled = settings.model.FALLBACK_MODEL
        self.load_model()

    def load_model(self) -> None:
        """Load the model from disk."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                if self.fallback_enabled:
                    logger.info("Using fallback model")
                    self._create_fallback_model()
                return

            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            # Load features if available
            features_path = self.model_path.replace(".pkl", "_features.json")
            if os.path.exists(features_path):
                with open(features_path, "r") as f:
                    self.features = json.load(f)
                logger.info(f"Loaded {len(self.features)} features from {features_path}")

            # Load metrics if available
            metrics_path = self.model_path.replace(".pkl", "_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {metrics_path}")

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if self.fallback_enabled:
                logger.info("Using fallback model")
                self._create_fallback_model()

    def _create_fallback_model(self) -> None:
        """Create a fallback model."""
        try:
            # Import necessary libraries
            from sklearn.ensemble import RandomForestRegressor

            # Create a simple model
            self.model = RandomForestRegressor(n_estimators=10, random_state=42)

            # Create dummy data and fit the model
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            self.model.fit(X, y)

            # Set default features
            self.features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

            logger.info("Fallback model created successfully")
        except Exception as e:
            logger.error(f"Error creating fallback model: {e}")
            self.model = None

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the model.

        Args:
            data: Input data

        Returns:
            Prediction results
        """
        if self.model is None:
            logger.error("No model available for prediction")
            return {"error": "No model available for prediction"}

        try:
            # Prepare input data
            X = self._prepare_input_data(data)

            # Make prediction
            prediction = self.model.predict(X)

            # Prepare response
            result = {
                "prediction": float(prediction[0]),
                "timestamp": datetime.now().isoformat()
            }

            # Add confidence interval if available
            if hasattr(self.model, "predict_interval"):
                lower, upper = self.model.predict_interval(X, alpha=0.05)
                result["confidence_interval"] = {
                    "lower": float(lower[0]),
                    "upper": float(upper[0])
                }

            # Add feature importance if available
            if hasattr(self.model, "feature_importances_") and self.features:
                importances = self.model.feature_importances_
                feature_importance = [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(self.features, importances)
                ]
                result["feature_importance"] = feature_importance

            return result
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": str(e)}

    def _prepare_input_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction.

        Args:
            data: Input data

        Returns:
            DataFrame with prepared input data
        """
        # Create DataFrame
        df = pd.DataFrame([data])

        # Check if features are available
        if self.features:
            # Check for missing features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    df[feature] = 0.0

            # Select only required features
            df = df[self.features]

        return df

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "features": self.features,
            "metrics": self.metrics,
            "fallback_enabled": self.fallback_enabled
        }

        # Add model type if available
        if self.model is not None:
            info["model_type"] = type(self.model).__name__

        return info


# Create model manager instance
model_manager = ModelManager()


@cached("prediction", ttl=60)
async def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction using the model.

    Args:
        data: Input data

    Returns:
        Prediction results
    """
    return model_manager.predict(data)
