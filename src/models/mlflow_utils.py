"""
MLflow utilities for experiment tracking.

This module provides functions for tracking experiments with MLflow,
including logging parameters, metrics, artifacts, and models.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Configure logging
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: Optional[str] = None, 
               experiment_name: str = "financial_forecasting") -> str:
    """
    Set up MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        
    Returns:
        Experiment ID
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment name: {experiment_name}")
    logger.info(f"MLflow experiment ID: {experiment_id}")
    
    return experiment_id


def start_run(run_name: Optional[str] = None, 
            experiment_name: str = "financial_forecasting",
            tracking_uri: Optional[str] = None,
            nested: bool = False) -> mlflow.ActiveRun:
    """
    Start an MLflow run.
    
    Args:
        run_name: Name of the run
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        nested: Whether this is a nested run
        
    Returns:
        MLflow active run
    """
    # Set up MLflow
    experiment_id = setup_mlflow(tracking_uri, experiment_name)
    
    # Start run
    run = mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        nested=nested
    )
    
    logger.info(f"Started MLflow run: {run.info.run_id}")
    
    return run


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameters
    """
    # Log each parameter
    for key, value in params.items():
        # Convert non-string values to string
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
        
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics
        step: Step number
    """
    # Log each metric
    for key, value in metrics.items():
        # Skip non-numeric values
        if not isinstance(value, (int, float)) or np.isnan(value):
            continue
        
        mlflow.log_metric(key, value, step=step)


def log_model(model: Any, model_name: str = "model", 
            X: Optional[pd.DataFrame] = None, 
            y: Optional[pd.Series] = None,
            conda_env: Optional[Dict[str, Any]] = None,
            registered_model_name: Optional[str] = None) -> None:
    """
    Log a model to MLflow.
    
    Args:
        model: Model to log
        model_name: Name of the model
        X: Feature matrix for signature inference
        y: Target vector for signature inference
        conda_env: Conda environment
        registered_model_name: Name to register the model under
    """
    # Infer signature if X and y are provided
    signature = None
    if X is not None and y is not None:
        signature = infer_signature(X, y)
    
    # Log model
    if hasattr(model, 'predict'):
        # Try to use specialized MLflow loggers for common model types
        model_type = type(model).__module__.split('.')[0]
        
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(
                model,
                model_name,
                signature=signature,
                conda_env=conda_env,
                registered_model_name=registered_model_name
            )
        elif model_type == 'sklearn':
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=signature,
                conda_env=conda_env,
                registered_model_name=registered_model_name
            )
        else:
            # Fall back to generic model logging
            mlflow.pyfunc.log_model(
                model_name,
                python_model=model,
                signature=signature,
                conda_env=conda_env,
                registered_model_name=registered_model_name
            )
    else:
        logger.warning(f"Model does not have a predict method, saving as artifact instead")
        
        # Save model to disk
        model_path = f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Log as artifact
        mlflow.log_artifact(model_path)
        
        # Remove temporary file
        os.remove(model_path)


def log_feature_importance(feature_names: List[str], importance: np.ndarray, 
                         top_n: int = 20) -> None:
    """
    Log feature importance to MLflow.
    
    Args:
        feature_names: List of feature names
        importance: Feature importance values
        top_n: Number of top features to log
    """
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Log as JSON
    top_features = importance_df.head(top_n).to_dict(orient='records')
    mlflow.log_dict(top_features, "feature_importance.json")
    
    # Create and log plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df.head(top_n)['feature'], importance_df.head(top_n)['importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()  # Display top features at the top
    
    # Save plot to disk
    plot_path = "feature_importance.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    
    # Log as artifact
    mlflow.log_artifact(plot_path)
    
    # Remove temporary file
    os.remove(plot_path)
    plt.close(fig)


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Log confusion matrix to MLflow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Log as JSON
    cm_dict = {
        'matrix': cm.tolist(),
        'labels': ['Negative', 'Positive']
    }
    mlflow.log_dict(cm_dict, "confusion_matrix.json")
    
    # Create and log plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Save plot to disk
    plot_path = "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    
    # Log as artifact
    mlflow.log_artifact(plot_path)
    
    # Remove temporary file
    os.remove(plot_path)
    plt.close(fig)


def log_predictions(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Log predictions to MLflow.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    """
    # Create DataFrame
    predictions_df = pd.DataFrame({
        'true': y_true.values,
        'predicted': y_pred
    })
    
    if isinstance(y_true, pd.Series) and y_true.index.name:
        predictions_df.index = y_true.index
        predictions_df.index.name = y_true.index.name
    
    # Save to disk
    predictions_path = "predictions.csv"
    predictions_df.to_csv(predictions_path)
    
    # Log as artifact
    mlflow.log_artifact(predictions_path)
    
    # Remove temporary file
    os.remove(predictions_path)
    
    # Create and log plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true.values, label='True')
    ax.plot(y_pred, label='Predicted')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('True vs Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot to disk
    plot_path = "predictions.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    
    # Log as artifact
    mlflow.log_artifact(plot_path)
    
    # Remove temporary file
    os.remove(plot_path)
    plt.close(fig)


def log_cv_results(cv_results: Dict[str, np.ndarray]) -> None:
    """
    Log cross-validation results to MLflow.
    
    Args:
        cv_results: Cross-validation results
    """
    # Log each metric
    for metric, values in cv_results.items():
        if metric.startswith('test_'):
            # Log mean and std
            mlflow.log_metric(f"{metric}_mean", float(np.mean(values)))
            mlflow.log_metric(f"{metric}_std", float(np.std(values)))
    
    # Log as JSON
    cv_results_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cv_results.items()}
    mlflow.log_dict(cv_results_dict, "cv_results.json")


def log_hyperparameter_tuning_results(results: Dict[str, Any]) -> None:
    """
    Log hyperparameter tuning results to MLflow.
    
    Args:
        results: Hyperparameter tuning results
    """
    # Log best parameters
    if 'best_params' in results:
        log_params(results['best_params'])
    
    # Log best score
    if 'best_score' in results:
        mlflow.log_metric('best_score', results['best_score'])
    
    # Log CV results
    if 'cv_results' in results:
        cv_results = results['cv_results']
        
        # Log mean test scores
        for key in cv_results:
            if key.startswith('mean_test_'):
                mlflow.log_metric(key, float(np.max(cv_results[key])))
        
        # Create and log plot for each parameter
        for param_name in results['best_params']:
            param_key = f'param_{param_name}'
            if param_key in cv_results:
                # Get parameter values
                param_values = cv_results[param_key]
                
                # Get mean test scores
                score_key = 'mean_test_score'
                if score_key in cv_results:
                    scores = cv_results[score_key]
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(param_values, scores, 'o-')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Score')
                    ax.set_title(f'Effect of {param_name} on Score')
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot to disk
                    plot_path = f"param_{param_name}.png"
                    fig.tight_layout()
                    fig.savefig(plot_path)
                    
                    # Log as artifact
                    mlflow.log_artifact(plot_path)
                    
                    # Remove temporary file
                    os.remove(plot_path)
                    plt.close(fig)
    
    # Log as JSON
    # Convert numpy arrays to lists
    results_dict = {}
    for k, v in results.items():
        if k == 'cv_results':
            results_dict[k] = {
                k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                for k2, v2 in v.items()
            }
        else:
            results_dict[k] = v
    
    mlflow.log_dict(results_dict, "tuning_results.json")


def log_dataset_info(X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: Optional[pd.DataFrame] = None, 
                   y_test: Optional[pd.Series] = None) -> None:
    """
    Log dataset information to MLflow.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Testing feature matrix
        y_test: Testing target vector
    """
    # Log dataset shapes
    mlflow.log_param('train_samples', X_train.shape[0])
    mlflow.log_param('features', X_train.shape[1])
    
    if X_test is not None:
        mlflow.log_param('test_samples', X_test.shape[0])
    
    # Log feature names
    mlflow.log_param('feature_names', list(X_train.columns))
    
    # Log target statistics
    mlflow.log_metric('target_mean', float(y_train.mean()))
    mlflow.log_metric('target_std', float(y_train.std()))
    mlflow.log_metric('target_min', float(y_train.min()))
    mlflow.log_metric('target_max', float(y_train.max()))
    
    # Create and log target distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_train, bins=50, alpha=0.7)
    if y_test is not None:
        ax.hist(y_test, bins=50, alpha=0.7)
        ax.legend(['Train', 'Test'])
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Target Distribution')
    ax.grid(True, alpha=0.3)
    
    # Save plot to disk
    plot_path = "target_distribution.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    
    # Log as artifact
    mlflow.log_artifact(plot_path)
    
    # Remove temporary file
    os.remove(plot_path)
    plt.close(fig)


def log_run_info(run_name: str, description: str, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Log run information to MLflow.
    
    Args:
        run_name: Name of the run
        description: Description of the run
        tags: Dictionary of tags
    """
    # Set run name
    mlflow.set_tag('mlflow.runName', run_name)
    
    # Log description
    mlflow.set_tag('mlflow.note.content', description)
    
    # Log tags
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()
    logger.info("Ended MLflow run")


def get_best_run(experiment_name: str, metric_name: str, 
               ascending: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Name of the metric to sort by
        ascending: Whether to sort in ascending order
        
    Returns:
        Dictionary with run information
    """
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.warning(f"Experiment {experiment_name} not found")
        return None
    
    # Get runs
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
    )
    
    if not runs:
        logger.warning(f"No runs found in experiment {experiment_name}")
        return None
    
    # Get best run
    best_run = runs[0]
    
    # Extract run information
    run_info = {
        'run_id': best_run.info.run_id,
        'run_name': best_run.data.tags.get('mlflow.runName', ''),
        'start_time': best_run.info.start_time,
        'end_time': best_run.info.end_time,
        'metrics': best_run.data.metrics,
        'params': best_run.data.params,
        'tags': best_run.data.tags
    }
    
    return run_info


def load_model_from_run(run_id: str, model_name: str = "model") -> Any:
    """
    Load a model from an MLflow run.
    
    Args:
        run_id: ID of the run
        model_name: Name of the model
        
    Returns:
        Loaded model
    """
    # Get run
    client = MlflowClient()
    run = client.get_run(run_id)
    
    # Get artifact URI
    artifact_uri = run.info.artifact_uri
    
    # Load model
    model_uri = f"{artifact_uri}/{model_name}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    return model


def track_experiment(experiment_name: str, run_name: str, 
                   model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   params: Dict[str, Any], metrics: Dict[str, float],
                   feature_names: List[str] = None,
                   description: str = "",
                   tags: Optional[Dict[str, str]] = None,
                   log_model_flag: bool = True,
                   registered_model_name: Optional[str] = None) -> str:
    """
    Track an experiment with MLflow.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        model: Model to log
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Testing feature matrix
        y_test: Testing target vector
        params: Dictionary of parameters
        metrics: Dictionary of metrics
        feature_names: List of feature names
        description: Description of the run
        tags: Dictionary of tags
        log_model_flag: Whether to log the model
        registered_model_name: Name to register the model under
        
    Returns:
        Run ID
    """
    # Start run
    with start_run(run_name=run_name, experiment_name=experiment_name) as run:
        # Log run info
        log_run_info(run_name, description, tags)
        
        # Log parameters
        log_params(params)
        
        # Log metrics
        log_metrics(metrics)
        
        # Log dataset info
        log_dataset_info(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log predictions
        log_predictions(y_test, y_pred)
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            
            log_feature_importance(feature_names, model.feature_importances_)
        
        # Log confusion matrix for classification
        if 'accuracy' in metrics or 'precision' in metrics or 'recall' in metrics:
            log_confusion_matrix(y_test, (y_pred > 0.5).astype(int))
        
        # Log model
        if log_model_flag:
            log_model(
                model,
                model_name="model",
                X=X_test,
                y=y_test,
                registered_model_name=registered_model_name
            )
        
        # Return run ID
        return run.info.run_id
