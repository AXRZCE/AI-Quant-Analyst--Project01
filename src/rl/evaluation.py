"""
Evaluation framework for reinforcement learning strategies.

This module provides tools for evaluating and comparing reinforcement learning
trading strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_performance_metrics(
    equity_curve: List[float],
    returns: List[float],
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve and returns.
    
    Args:
        equity_curve: List of equity values
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy arrays
    equity = np.array(equity_curve)
    rets = np.array(returns)
    
    # Skip first return (usually 0)
    rets = rets[1:]
    
    # Calculate basic metrics
    total_return = equity[-1] / equity[0] - 1
    
    # Annualized metrics (assuming daily returns)
    n_days = len(rets)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    daily_vol = np.std(rets)
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe ratio
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = rets - daily_risk_free
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    # Sortino ratio
    downside_returns = rets[rets < 0]
    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Win rate
    win_rate = np.mean(rets > 0) if len(rets) > 0 else 0
    
    # Profit factor
    gross_profits = np.sum(rets[rets > 0]) if len(rets[rets > 0]) > 0 else 0
    gross_losses = np.abs(np.sum(rets[rets < 0])) if len(rets[rets < 0]) > 0 else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Return metrics
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor
    }


def evaluate_strategy(
    strategy,
    test_df: pd.DataFrame,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluate a trading strategy.
    
    Args:
        strategy: Trading strategy to evaluate
        test_df: Test DataFrame
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with evaluation results
    """
    # Run backtest
    backtest_results = strategy.backtest(
        df=test_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        equity_curve=backtest_results["equity_curve"],
        returns=backtest_results["returns"],
        risk_free_rate=risk_free_rate
    )
    
    # Combine results
    results = {
        "backtest_results": backtest_results,
        "metrics": metrics
    }
    
    return results


def compare_strategies(
    strategies: Dict[str, Any],
    test_df: pd.DataFrame,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Compare multiple trading strategies.
    
    Args:
        strategies: Dictionary of strategy name to strategy
        test_df: Test DataFrame
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of price
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with comparison results
    """
    # Evaluate each strategy
    results = {}
    
    for name, strategy in strategies.items():
        logger.info(f"Evaluating strategy: {name}")
        
        # Evaluate strategy
        eval_results = evaluate_strategy(
            strategy=strategy,
            test_df=test_df,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            slippage=slippage,
            risk_free_rate=risk_free_rate
        )
        
        # Store results
        results[name] = eval_results
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        name: results[name]["metrics"]
        for name in strategies.keys()
    })
    
    # Transpose for better readability
    comparison = comparison.T
    
    # Add to results
    results["comparison"] = comparison
    
    return results


def plot_comparison(
    comparison_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    metrics: Optional[List[str]] = None
) -> None:
    """
    Plot strategy comparison.
    
    Args:
        comparison_results: Results from compare_strategies
        figsize: Figure size
        metrics: List of metrics to plot (if None, plot all)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
        
        # Get strategies
        strategies = [name for name in comparison_results.keys() if name != "comparison"]
        
        # Create figure for equity curves
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curves
        for name in strategies:
            equity = comparison_results[name]["backtest_results"]["equity_curve"]
            dates = pd.to_datetime(comparison_results[name]["backtest_results"]["dates"])
            
            ax.plot(dates, equity, label=name)
        
        # Set labels and title
        ax.set_title("Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.grid(True)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Create figure for metrics comparison
        if metrics is None:
            # Use all metrics except those with very large values
            all_metrics = comparison_results["comparison"].columns.tolist()
            metrics = [m for m in all_metrics if m not in ["profit_factor"]]
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get values
            values = comparison_results["comparison"][metric]
            
            # Create bar chart
            ax.bar(values.index, values.values)
            
            # Set labels and title
            ax.set_title(metric)
            ax.set_ylabel("Value")
            ax.grid(True, axis="y")
            
            # Format y-axis for percentage metrics
            if metric in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"]:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available. Install with 'pip install matplotlib'")
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")
