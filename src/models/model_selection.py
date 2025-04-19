"""
Model selection framework.

This module provides functionality for selecting the best model from a set of
candidate models, including model registry, model comparison, and model selection.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for tracking and comparing models."""
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory for storing model registry
        """
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self.registry = self._load_registry()
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry.
        
        Returns:
            Model registry dictionary
        """
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                logger.info(f"Loaded model registry with {len(registry.get('models', []))} models")
                return registry
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                return {"models": []}
        else:
            logger.info("Model registry not found, creating new registry")
            return {"models": []}
    
    def _save_registry(self) -> None:
        """Save the model registry."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Saved model registry with {len(self.registry.get('models', []))} models")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Register a model in the registry.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to the model file
            model_type: Type of model
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata
            tags: List of tags
            
        Returns:
            Model entry
        """
        # Check if model already exists
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                logger.warning(f"Model {model_id} already exists in registry, updating")
                model.update({
                    "model_path": model_path,
                    "model_type": model_type,
                    "metrics": metrics,
                    "metadata": metadata or {},
                    "tags": tags or [],
                    "updated_at": time.time()
                })
                self._save_registry()
                return model
        
        # Create new model entry
        model_entry = {
            "model_id": model_id,
            "model_path": model_path,
            "model_type": model_type,
            "metrics": metrics,
            "metadata": metadata or {},
            "tags": tags or [],
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Add to registry
        self.registry["models"].append(model_entry)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {model_id} in registry")
        
        return model_entry
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model entry or None if not found
        """
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        
        logger.warning(f"Model {model_id} not found in registry")
        return None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry.
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags
            sort_by: Sort by field
            ascending: Sort in ascending order
            
        Returns:
            List of model entries
        """
        # Filter models
        models = self.registry["models"]
        
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        if tags:
            models = [m for m in models if all(tag in m["tags"] for tag in tags)]
        
        # Sort models
        if sort_by:
            if sort_by.startswith("metrics."):
                # Sort by metric
                metric_name = sort_by.split(".", 1)[1]
                models = sorted(
                    models,
                    key=lambda m: m["metrics"].get(metric_name, float('inf') if ascending else float('-inf')),
                    reverse=not ascending
                )
            else:
                # Sort by field
                models = sorted(
                    models,
                    key=lambda m: m.get(sort_by, ""),
                    reverse=not ascending
                )
        
        return models
    
    def get_best_model(
        self,
        metric: str,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ascending: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            model_type: Filter by model type
            tags: Filter by tags
            ascending: Whether lower is better
            
        Returns:
            Best model entry or None if no models found
        """
        # List models
        models = self.list_models(
            model_type=model_type,
            tags=tags,
            sort_by=f"metrics.{metric}",
            ascending=ascending
        )
        
        if not models:
            logger.warning(f"No models found for criteria: metric={metric}, model_type={model_type}, tags={tags}")
            return None
        
        return models[0]
    
    def load_model(self, model_id: str) -> Tuple[Optional[BaseEstimator], Optional[Dict[str, Any]]]:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, metadata) or (None, None) if not found
        """
        # Get model entry
        model_entry = self.get_model(model_id)
        
        if not model_entry:
            return None, None
        
        # Load model
        try:
            model_path = model_entry["model_path"]
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None, None
            
            # Load model
            model_package = joblib.load(model_path)
            
            # Extract model and metadata
            if isinstance(model_package, dict) and "model" in model_package:
                model = model_package["model"]
                metadata = model_package.get("metadata", {})
            else:
                # If not a package, assume it's the model directly
                model = model_package
                metadata = {}
            
            logger.info(f"Loaded model {model_id} from {model_path}")
            
            return model, metadata
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None, None
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False otherwise
        """
        # Get model entry
        model_entry = self.get_model(model_id)
        
        if not model_entry:
            return False
        
        # Remove from registry
        self.registry["models"] = [m for m in self.registry["models"] if m["model_id"] != model_id]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted model {model_id} from registry")
        
        return True
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare models based on metrics.
        
        Args:
            model_ids: List of model identifiers
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with model comparison
        """
        # Get model entries
        models = [self.get_model(model_id) for model_id in model_ids]
        models = [m for m in models if m is not None]
        
        if not models:
            logger.warning("No models found for comparison")
            return pd.DataFrame()
        
        # Get all metrics if not specified
        if not metrics:
            metrics = set()
            for model in models:
                metrics.update(model["metrics"].keys())
            metrics = sorted(metrics)
        
        # Create comparison DataFrame
        data = []
        for model in models:
            row = {
                "model_id": model["model_id"],
                "model_type": model["model_type"],
                "created_at": model["created_at"]
            }
            
            # Add metrics
            for metric in metrics:
                row[metric] = model["metrics"].get(metric, None)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_model_comparison(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            model_ids: List of model identifiers
            metrics: List of metrics to compare
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get comparison DataFrame
        df = self.compare_models(model_ids, metrics)
        
        if df.empty:
            logger.warning("No models found for comparison")
            return plt.figure()
        
        # Get metrics
        metrics = [col for col in df.columns if col not in ["model_id", "model_type", "created_at"]]
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Sort by metric
            df_sorted = df.sort_values(metric)
            
            # Plot metric
            ax.barh(df_sorted["model_id"], df_sorted[metric])
            ax.set_title(metric)
            ax.set_xlabel("Value")
            ax.set_ylabel("Model")
            
            # Add values
            for j, v in enumerate(df_sorted[metric]):
                ax.text(v, j, f" {v:.4f}", va="center")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class ModelSelector:
    """Model selector for selecting the best model."""
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        registry_dir: str = "models/registry",
        primary_metric: str = "rmse",
        lower_is_better: bool = True
    ):
        """
        Initialize the model selector.
        
        Args:
            registry: Model registry
            registry_dir: Directory for model registry
            primary_metric: Primary metric for model selection
            lower_is_better: Whether lower metric values are better
        """
        self.registry = registry or ModelRegistry(registry_dir)
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better
    
    def select_best_model(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[Optional[BaseEstimator], Optional[Dict[str, Any]]]:
        """
        Select the best model based on the primary metric.
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags
            
        Returns:
            Tuple of (best model, metadata) or (None, None) if no models found
        """
        # Get best model entry
        best_model_entry = self.registry.get_best_model(
            metric=self.primary_metric,
            model_type=model_type,
            tags=tags,
            ascending=self.lower_is_better
        )
        
        if not best_model_entry:
            logger.warning(f"No models found for criteria: metric={self.primary_metric}, model_type={model_type}, tags={tags}")
            return None, None
        
        # Load best model
        model, metadata = self.registry.load_model(best_model_entry["model_id"])
        
        if model is None:
            logger.error(f"Failed to load best model {best_model_entry['model_id']}")
            return None, None
        
        logger.info(f"Selected best model {best_model_entry['model_id']} with {self.primary_metric}={best_model_entry['metrics'].get(self.primary_metric)}")
        
        return model, metadata
    
    def evaluate_and_register_model(
        self,
        model: BaseEstimator,
        model_id: str,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate and register a model.
        
        Args:
            model: Model to evaluate and register
            model_id: Unique model identifier
            model_type: Type of model
            X: Training feature matrix
            y: Training target vector
            X_val: Validation feature matrix
            y_val: Validation target vector
            metrics: Dictionary of metric names and functions
            metadata: Additional metadata
            tags: List of tags
            save_path: Path to save the model
            
        Returns:
            Model entry
        """
        # Use default metrics if not provided
        if metrics is None:
            metrics = {
                'mse': mean_squared_error,
                'mae': mean_absolute_error,
                'r2': r2_score
            }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            logger.info(f"Evaluating model {model_id} on validation set")
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            eval_metrics = {}
            for name, metric_fn in metrics.items():
                eval_metrics[name] = float(metric_fn(y_val, y_pred))
            
            # Add RMSE
            if 'mse' in eval_metrics:
                eval_metrics['rmse'] = float(np.sqrt(eval_metrics['mse']))
        
        # Otherwise, use cross-validation
        else:
            logger.info(f"Evaluating model {model_id} using cross-validation")
            
            # Create cross-validation object
            if isinstance(X, pd.DataFrame) and 'timestamp' in X.columns:
                # Use time series split for time series data
                cv = TimeSeriesSplit(n_splits=5)
            else:
                # Use regular K-fold for other data
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Calculate metrics using cross-validation
            eval_metrics = {}
            for name, metric_fn in metrics.items():
                # Use negative for metrics where higher is better
                if name == 'r2':
                    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                    eval_metrics[name] = float(np.mean(scores))
                else:
                    # For error metrics, lower is better
                    scores = []
                    for train_idx, val_idx in cv.split(X, y):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Fit model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_val)
                        
                        # Calculate metric
                        score = metric_fn(y_val, y_pred)
                        scores.append(score)
                    
                    eval_metrics[name] = float(np.mean(scores))
            
            # Add RMSE
            if 'mse' in eval_metrics:
                eval_metrics['rmse'] = float(np.sqrt(eval_metrics['mse']))
        
        # Log metrics
        for name, value in eval_metrics.items():
            logger.info(f"{name}: {value:.6f}")
        
        # Save model if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            model_package = {
                "model": model,
                "metadata": metadata or {}
            }
            
            joblib.dump(model_package, save_path)
            logger.info(f"Saved model to {save_path}")
        else:
            # Generate default save path
            save_path = os.path.join(self.registry.registry_dir, f"{model_id}.pkl")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            model_package = {
                "model": model,
                "metadata": metadata or {}
            }
            
            joblib.dump(model_package, save_path)
            logger.info(f"Saved model to {save_path}")
        
        # Register model
        model_entry = self.registry.register_model(
            model_id=model_id,
            model_path=save_path,
            model_type=model_type,
            metrics=eval_metrics,
            metadata=metadata,
            tags=tags
        )
        
        return model_entry
    
    def is_better_model(
        self,
        model_id: str,
        reference_model_id: Optional[str] = None
    ) -> bool:
        """
        Check if a model is better than a reference model.
        
        Args:
            model_id: Model identifier
            reference_model_id: Reference model identifier (if None, use best model)
            
        Returns:
            True if the model is better, False otherwise
        """
        # Get model entry
        model_entry = self.registry.get_model(model_id)
        
        if not model_entry:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        # Get reference model entry
        if reference_model_id:
            reference_model_entry = self.registry.get_model(reference_model_id)
            
            if not reference_model_entry:
                logger.warning(f"Reference model {reference_model_id} not found in registry")
                return True  # No reference model, so this model is better
        else:
            # Use best model as reference
            reference_model_entry = self.registry.get_best_model(
                metric=self.primary_metric,
                ascending=self.lower_is_better
            )
            
            if not reference_model_entry or reference_model_entry["model_id"] == model_id:
                return True  # No reference model or this is the best model
        
        # Compare metrics
        model_metric = model_entry["metrics"].get(self.primary_metric)
        reference_metric = reference_model_entry["metrics"].get(self.primary_metric)
        
        if model_metric is None or reference_metric is None:
            logger.warning(f"Missing metric {self.primary_metric} for comparison")
            return False
        
        # Check if better
        if self.lower_is_better:
            return model_metric < reference_metric
        else:
            return model_metric > reference_metric
