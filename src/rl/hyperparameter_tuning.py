"""
Hyperparameter tuning for reinforcement learning strategies.

This module provides tools for tuning hyperparameters of reinforcement learning
trading strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
import itertools
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from src.rl.evaluation import evaluate_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def grid_search(
    strategy_class,
    param_grid: Dict[str, List[Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    maximize: bool = True,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0,
    n_trials: Optional[int] = None,
    random_search: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform grid search for hyperparameter tuning.

    Args:
        strategy_class: Strategy class to tune
        param_grid: Dictionary of parameter name to list of values
        train_df: Training DataFrame
        val_df: Validation DataFrame
        metric: Metric to optimize
        maximize: Whether to maximize or minimize the metric
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        n_trials: Number of trials for random search (if None, try all combinations)
        random_search: Whether to use random search instead of grid search
        verbose: Whether to print progress

    Returns:
        Dictionary with tuning results
    """
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    if random_search:
        # Random search
        if n_trials is None:
            # Calculate total number of combinations
            n_combinations = 1
            for values in param_values:
                n_combinations *= len(values)

            # Set n_trials to 10% of total combinations or 10, whichever is larger
            n_trials = max(10, int(n_combinations * 0.1))

        # Generate random combinations
        param_combinations = []
        for _ in range(n_trials):
            combination = [random.choice(values) for values in param_values]
            param_combinations.append(combination)
    else:
        # Grid search
        param_combinations = list(itertools.product(*param_values))

        # Limit number of trials if specified
        if n_trials is not None:
            if n_trials < len(param_combinations):
                # Randomly sample combinations
                param_combinations = random.sample(param_combinations, n_trials)

    # Initialize results
    results = []
    best_score = float('-inf') if maximize else float('inf')
    best_params = None
    best_strategy = None

    # Evaluate each combination
    for i, combination in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))

        if verbose:
            logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")

        try:
            # Create strategy
            strategy = strategy_class(**params)

            # Train strategy
            if hasattr(strategy, 'train'):
                strategy.train(train_df)

            # Evaluate strategy
            eval_results = evaluate_strategy(
                strategy=strategy,
                test_df=val_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage,
                risk_free_rate=risk_free_rate
            )

            # Get score
            score = eval_results["metrics"][metric]

            # Check if best
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params
                best_strategy = strategy

            # Add to results
            results.append({
                "params": params,
                "score": score,
                "metrics": eval_results["metrics"]
            })

            if verbose:
                logger.info(f"Score: {score:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")

            # Add to results
            results.append({
                "params": params,
                "score": float('-inf') if maximize else float('inf'),
                "metrics": None,
                "error": str(e)
            })

    # Sort results
    results.sort(key=lambda x: x["score"], reverse=maximize)

    # Create results dictionary
    tuning_results = {
        "best_params": best_params,
        "best_score": best_score,
        "best_strategy": best_strategy,
        "results": results
    }

    return tuning_results


def bayesian_optimization(
    strategy_class,
    param_space: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    maximize: bool = True,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0,
    n_trials: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization for hyperparameter tuning.

    Args:
        strategy_class: Strategy class to tune
        param_space: Dictionary of parameter name to parameter space
        train_df: Training DataFrame
        val_df: Validation DataFrame
        metric: Metric to optimize
        maximize: Whether to maximize or minimize the metric
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        n_trials: Number of trials
        verbose: Whether to print progress

    Returns:
        Dictionary with tuning results
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
    except ImportError:
        logger.error("scikit-optimize not installed. Install with 'pip install scikit-optimize'")
        raise

    # Convert param_space to skopt space
    space = []
    param_names = []

    for name, param in param_space.items():
        param_names.append(name)

        if isinstance(param, tuple) and len(param) == 3:
            # (low, high, type)
            low, high, param_type = param

            if param_type == 'int':
                space.append(Integer(low, high, name=name))
            elif param_type == 'float':
                space.append(Real(low, high, name=name))
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        elif isinstance(param, list):
            # Categorical parameter
            space.append(Categorical(param, name=name))

        else:
            raise ValueError(f"Invalid parameter space for {name}: {param}")

    # Define objective function
    @use_named_args(space)
    def objective(**params):
        if verbose:
            logger.info(f"Evaluating parameters: {params}")

        try:
            # Create strategy
            strategy = strategy_class(**params)

            # Train strategy
            if hasattr(strategy, 'train'):
                strategy.train(train_df)

            # Evaluate strategy
            eval_results = evaluate_strategy(
                strategy=strategy,
                test_df=val_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage,
                risk_free_rate=risk_free_rate
            )

            # Get score
            score = eval_results["metrics"][metric]

            if verbose:
                logger.info(f"Score: {score:.4f}")

            # Return negative score if maximizing (gp_minimize minimizes)
            return -score if maximize else score

        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")

            # Return worst possible score
            return float('inf')

    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_trials,
        random_state=42,
        verbose=verbose
    )

    # Get best parameters
    best_params = dict(zip(param_names, result.x))

    # Create strategy with best parameters
    best_strategy = strategy_class(**best_params)

    # Train strategy
    if hasattr(best_strategy, 'train'):
        best_strategy.train(train_df)

    # Evaluate strategy
    eval_results = evaluate_strategy(
        strategy=best_strategy,
        test_df=val_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
        risk_free_rate=risk_free_rate
    )

    # Get best score
    best_score = eval_results["metrics"][metric]

    # Create results list
    results = []

    for i, (x, func_val) in enumerate(zip(result.x_iters, result.func_vals)):
        params = dict(zip(param_names, x))
        score = -func_val if maximize else func_val

        results.append({
            "params": params,
            "score": score
        })

    # Sort results
    results.sort(key=lambda x: x["score"], reverse=maximize)

    # Create results dictionary
    tuning_results = {
        "best_params": best_params,
        "best_score": best_score,
        "best_strategy": best_strategy,
        "results": results
    }

    return tuning_results


def cross_validate(
    strategy_class,
    params: Dict[str, Any],
    df: pd.DataFrame,
    n_splits: int = 5,
    metric: str = "sharpe_ratio",
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation for a strategy.

    Args:
        strategy_class: Strategy class to validate
        params: Strategy parameters
        df: DataFrame with data
        n_splits: Number of splits for cross-validation
        metric: Metric to evaluate
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        verbose: Whether to print progress

    Returns:
        Dictionary with cross-validation results
    """
    # Get unique dates
    dates = df["timestamp"].unique()
    # Sort dates (convert to numpy array first if it's a pandas DatetimeArray)
    dates = np.sort(dates)

    # Calculate split size
    split_size = len(dates) // n_splits

    # Initialize results
    scores = []
    metrics_list = []

    # Perform cross-validation
    for i in range(n_splits):
        if verbose:
            logger.info(f"Fold {i+1}/{n_splits}")

        # Calculate split indices
        val_start = i * split_size
        val_end = (i + 1) * split_size if i < n_splits - 1 else len(dates)

        # Split data
        val_dates = dates[val_start:val_end]
        train_dates = np.concatenate([dates[:val_start], dates[val_end:]])

        train_df = df[df["timestamp"].isin(train_dates)]
        val_df = df[df["timestamp"].isin(val_dates)]

        try:
            # Create strategy
            strategy = strategy_class(**params)

            # Train strategy
            if hasattr(strategy, 'train'):
                strategy.train(train_df)

            # Evaluate strategy
            eval_results = evaluate_strategy(
                strategy=strategy,
                test_df=val_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage,
                risk_free_rate=risk_free_rate
            )

            # Get score
            score = eval_results["metrics"][metric]

            # Add to results
            scores.append(score)
            metrics_list.append(eval_results["metrics"])

            if verbose:
                logger.info(f"Score: {score:.4f}")

        except Exception as e:
            logger.error(f"Error in fold {i+1}: {e}")

            # Add NaN to results
            scores.append(float('nan'))
            metrics_list.append(None)

    # Calculate mean and std of scores
    mean_score = np.nanmean(scores)
    std_score = np.nanstd(scores)

    # Calculate mean metrics
    mean_metrics = {}

    for metric_name in metrics_list[0].keys():
        values = [m[metric_name] for m in metrics_list if m is not None and metric_name in m]
        mean_metrics[metric_name] = np.nanmean(values)

    # Create results dictionary
    cv_results = {
        "params": params,
        "mean_score": mean_score,
        "std_score": std_score,
        "scores": scores,
        "mean_metrics": mean_metrics,
        "metrics": metrics_list
    }

    return cv_results


def walk_forward_validation(
    strategy_class,
    params: Dict[str, Any],
    df: pd.DataFrame,
    train_window: int = 252,  # 1 year of trading days
    val_window: int = 63,     # 3 months of trading days
    step: int = 21,           # 1 month of trading days
    metric: str = "sharpe_ratio",
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform walk-forward validation for a strategy.

    Args:
        strategy_class: Strategy class to validate
        params: Strategy parameters
        df: DataFrame with data
        train_window: Number of days in training window
        val_window: Number of days in validation window
        step: Number of days to step forward
        metric: Metric to evaluate
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        verbose: Whether to print progress

    Returns:
        Dictionary with walk-forward validation results
    """
    # Get unique dates
    dates = df["timestamp"].unique()
    # Sort dates (convert to numpy array first if it's a pandas DatetimeArray)
    dates = np.sort(dates)

    # Check if enough data
    if len(dates) < train_window + val_window:
        raise ValueError(f"Not enough data for walk-forward validation. Need at least {train_window + val_window} days, got {len(dates)}")

    # Initialize results
    results = []

    # Perform walk-forward validation
    for i in range(0, len(dates) - train_window - val_window + 1, step):
        # Calculate window indices
        train_start = i
        train_end = i + train_window
        val_start = train_end
        val_end = val_start + val_window

        # Get window dates
        train_dates = dates[train_start:train_end]
        val_dates = dates[val_start:val_end]

        if verbose:
            logger.info(f"Window {i//step + 1}: Training {train_dates[0]} to {train_dates[-1]}, Validation {val_dates[0]} to {val_dates[-1]}")

        # Split data
        train_df = df[df["timestamp"].isin(train_dates)]
        val_df = df[df["timestamp"].isin(val_dates)]

        try:
            # Create strategy
            strategy = strategy_class(**params)

            # Train strategy
            if hasattr(strategy, 'train'):
                strategy.train(train_df)

            # Evaluate strategy
            eval_results = evaluate_strategy(
                strategy=strategy,
                test_df=val_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage,
                risk_free_rate=risk_free_rate
            )

            # Get score
            score = eval_results["metrics"][metric]

            # Add to results
            results.append({
                "window": i//step + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "val_start": val_dates[0],
                "val_end": val_dates[-1],
                "score": score,
                "metrics": eval_results["metrics"]
            })

            if verbose:
                logger.info(f"Score: {score:.4f}")

        except Exception as e:
            logger.error(f"Error in window {i//step + 1}: {e}")

            # Add to results with error
            results.append({
                "window": i//step + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "val_start": val_dates[0],
                "val_end": val_dates[-1],
                "score": float('nan'),
                "metrics": None,
                "error": str(e)
            })

    # Calculate mean and std of scores
    scores = [r["score"] for r in results if "error" not in r]
    mean_score = np.nanmean(scores)
    std_score = np.nanstd(scores)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Create results dictionary
    wf_results = {
        "params": params,
        "mean_score": mean_score,
        "std_score": std_score,
        "results": results,
        "results_df": results_df
    }

    return wf_results


def plot_walk_forward_results(
    wf_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot walk-forward validation results.

    Args:
        wf_results: Results from walk_forward_validation
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        # Get results DataFrame
        results_df = wf_results["results_df"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot scores
        ax.plot(results_df["val_end"], results_df["score"], marker='o')

        # Set labels and title
        ax.set_title("Walk-Forward Validation Results")
        ax.set_xlabel("Validation End Date")
        ax.set_ylabel("Score")
        ax.grid(True)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add mean score line
        ax.axhline(wf_results["mean_score"], color='r', linestyle='--', label=f"Mean Score: {wf_results['mean_score']:.4f}")

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available. Install with 'pip install matplotlib'")
    except Exception as e:
        logger.error(f"Error plotting walk-forward results: {e}")
