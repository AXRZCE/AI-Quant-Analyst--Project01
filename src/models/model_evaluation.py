"""
Model evaluation module for financial time series.

This module provides functions for evaluating models on financial time series data,
including standard regression metrics and financial-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Configure logging
logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Median Absolute Error
    metrics['median_ae'] = median_absolute_error(y_true, y_pred)
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Explained Variance
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # Correlation
    if len(y_true) > 1:  # Correlation requires at least 2 samples
        metrics['pearson_corr'], metrics['pearson_pval'] = pearsonr(y_true, y_pred)
        metrics['spearman_corr'], metrics['spearman_pval'] = spearmanr(y_true, y_pred)
    
    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate standard classification metrics.
    
    Args:
        y_true: True target values (0 or 1)
        y_pred: Predicted target values (0 or 1)
        y_prob: Predicted probabilities (for ROC AUC)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC (if probabilities are provided)
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    return metrics


def calculate_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              prices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate financial-specific metrics.
    
    Args:
        y_true: True target values (returns)
        y_pred: Predicted target values (returns)
        prices: Price series (optional, for calculating PnL)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Directional Accuracy (% of times the direction is correctly predicted)
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
    
    # Information Coefficient (Spearman rank correlation)
    if len(y_true) > 1:  # Correlation requires at least 2 samples
        metrics['ic'], _ = spearmanr(y_true, y_pred)
    else:
        metrics['ic'] = np.nan
    
    # Maximum Drawdown of prediction errors
    errors = np.abs(y_true - y_pred)
    cumulative_errors = np.cumsum(errors)
    max_cumulative_errors = np.maximum.accumulate(cumulative_errors)
    drawdowns = max_cumulative_errors - cumulative_errors
    metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Profit Factor (sum of positive returns / sum of negative returns)
    if prices is not None and len(prices) > 1:
        # Calculate returns from prices
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate predicted positions (1 for long, -1 for short, 0 for no position)
        positions = np.sign(y_pred[:-1])
        
        # Calculate PnL
        pnl = positions * returns
        
        # Calculate profit factor
        positive_pnl = pnl[pnl > 0].sum()
        negative_pnl = -pnl[pnl < 0].sum()
        
        if negative_pnl > 0:
            metrics['profit_factor'] = positive_pnl / negative_pnl
        else:
            metrics['profit_factor'] = np.inf if positive_pnl > 0 else 0
        
        # Calculate Sharpe ratio
        if len(pnl) > 1:
            metrics['sharpe_ratio'] = np.mean(pnl) / np.std(pnl) * np.sqrt(252)  # Annualized
        else:
            metrics['sharpe_ratio'] = np.nan
        
        # Calculate Win Rate
        metrics['win_rate'] = np.mean(pnl > 0)
        
        # Calculate Average Win/Loss Ratio
        avg_win = np.mean(pnl[pnl > 0]) if np.any(pnl > 0) else 0
        avg_loss = np.mean(pnl[pnl < 0]) if np.any(pnl < 0) else 0
        
        if avg_loss != 0:
            metrics['win_loss_ratio'] = abs(avg_win / avg_loss)
        else:
            metrics['win_loss_ratio'] = np.inf if avg_win > 0 else 0
    
    return metrics


def evaluate_regression_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                            prices: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Evaluate a regression model on financial time series data.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: Target vector (returns)
        prices: Price series (optional, for calculating PnL)
        
    Returns:
        Dictionary of metric names and values
    """
    logger.info("Evaluating regression model")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate standard regression metrics
    metrics = calculate_regression_metrics(y.values, y_pred)
    
    # Calculate financial metrics
    financial_metrics = calculate_financial_metrics(
        y.values, y_pred, 
        prices.values if prices is not None else None
    )
    
    # Combine metrics
    metrics.update(financial_metrics)
    
    # Log metrics
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.6f}")
    
    return metrics


def evaluate_classification_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate a classification model on financial time series data.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X: Feature matrix
        y: Target vector (0 or 1)
        
    Returns:
        Dictionary of metric names and values
    """
    logger.info("Evaluating classification model")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Make probability predictions if possible
    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
    
    # Calculate classification metrics
    metrics = calculate_classification_metrics(y.values, y_pred, y_prob)
    
    # Log metrics
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{name}: {value:.6f}")
    
    return metrics


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, title: str = "Actual vs Predicted") -> plt.Figure:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.plot(y_true.index, y_true.values, label="Actual", color="blue", alpha=0.7)
    
    # Plot predicted values
    ax.plot(y_true.index, y_pred, label="Predicted", color="red", alpha=0.7)
    
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, 
                          title: str = "Feature Importance") -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    sns.barplot(x="importance", y="feature", data=top_features, ax=ax)
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True target values (0 or 1)
        y_pred: Predicted target values (0 or 1)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    
    return fig


def plot_cumulative_returns(returns: pd.Series, title: str = "Cumulative Returns") -> plt.Figure:
    """
    Plot cumulative returns.
    
    Args:
        returns: Series of returns
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative returns
    ax.plot(cumulative_returns.index, cumulative_returns.values, color="green")
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    
    return fig


def plot_model_performance(y_true: pd.Series, y_pred: np.ndarray, 
                         prices: Optional[pd.Series] = None) -> Dict[str, plt.Figure]:
    """
    Plot various model performance charts.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        prices: Price series (optional)
        
    Returns:
        Dictionary of plot names and figures
    """
    plots = {}
    
    # Plot actual vs predicted
    plots["actual_vs_predicted"] = plot_predictions(y_true, y_pred)
    
    # Plot prediction errors
    errors = y_true - y_pred
    plots["prediction_errors"] = plot_predictions(
        pd.Series(errors, index=y_true.index), 
        np.zeros_like(errors),
        title="Prediction Errors"
    )
    
    # Plot error distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, ax=ax)
    ax.set_title("Error Distribution")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    plots["error_distribution"] = fig
    
    # Plot cumulative returns if prices are provided
    if prices is not None and len(prices) > 1:
        # Calculate returns from prices
        returns = prices.pct_change().dropna()
        
        # Calculate predicted positions (1 for long, -1 for short, 0 for no position)
        positions = pd.Series(np.sign(y_pred), index=y_true.index)
        positions = positions.shift(1).dropna()  # Shift to avoid lookahead bias
        
        # Calculate strategy returns
        strategy_returns = positions * returns.loc[positions.index]
        
        # Plot cumulative returns
        plots["cumulative_returns"] = plot_cumulative_returns(
            pd.concat([returns, strategy_returns], axis=1).dropna(),
            title="Cumulative Returns: Buy & Hold vs Strategy"
        )
    
    return plots


def generate_evaluation_report(model: Any, X: pd.DataFrame, y: pd.Series, 
                             feature_names: List[str], 
                             prices: Optional[pd.Series] = None,
                             is_classification: bool = False) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        prices: Price series (optional)
        is_classification: Whether the model is a classification model
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Generating evaluation report")
    
    report = {}
    
    # Evaluate model
    if is_classification:
        report["metrics"] = evaluate_classification_model(model, X, y)
    else:
        report["metrics"] = evaluate_regression_model(model, X, y, prices)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        report["feature_importance"] = importance_df
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Generate plots
    report["plots"] = plot_model_performance(y, y_pred, prices)
    
    # Add confusion matrix for classification models
    if is_classification:
        report["plots"]["confusion_matrix"] = plot_confusion_matrix(y, y_pred)
    
    # Add feature importance plot if available
    if "feature_importance" in report:
        report["plots"]["feature_importance"] = plot_feature_importance(report["feature_importance"])
    
    return report
