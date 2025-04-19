"""
Tests for the reinforcement learning hyperparameter tuning framework.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl.hyperparameter_tuning import (
    grid_search,
    cross_validate,
    walk_forward_validation
)


class DummyStrategy:
    """Dummy strategy for testing."""
    
    def __init__(self, param1=0.1, param2=10, param3="value1"):
        """Initialize dummy strategy."""
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        
        # Calculate a deterministic score based on parameters
        self.score = param1 * 10 + param2 * 0.01 + (1 if param3 == "value1" else 0)
    
    def train(self, df):
        """Train dummy strategy."""
        pass
    
    def backtest(self, df, initial_capital=100_000, transaction_cost=0.001, slippage=0.0005):
        """Run dummy backtest."""
        # Get unique dates
        dates = df["timestamp"].unique()
        
        # Generate dummy equity curve and returns
        equity_curve = [initial_capital]
        returns = [0.0]
        
        for i in range(len(dates)):
            # Use deterministic return based on parameters
            ret = self.score * 0.001
            
            # Add some noise
            ret += np.random.normal(0, 0.001)
            
            # Calculate new equity
            new_equity = equity_curve[-1] * (1 + ret)
            
            # Add to lists
            equity_curve.append(new_equity)
            returns.append(ret)
        
        # Generate dummy trades
        trades = []
        for i in range(10):
            trades.append({
                "date": dates[i % len(dates)],
                "symbol": "AAPL",
                "action": "buy" if i % 2 == 0 else "sell",
                "price": 150.0 + np.random.randn() * 5.0,
                "size": 10.0,
                "value": 1500.0,
                "cost": 1.5
            })
        
        # Return results
        return {
            "initial_capital": initial_capital,
            "final_equity": equity_curve[-1],
            "total_return": equity_curve[-1] / initial_capital - 1,
            "equity_curve": equity_curve,
            "returns": returns,
            "dates": [dates[0] - timedelta(days=1)] + list(dates),
            "trades": trades
        }


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    symbols = ['AAPL', 'MSFT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            price = 150.0 + np.random.randn() * 5.0 if symbol == 'AAPL' else 250.0 + np.random.randn() * 8.0
            volume = 1000000 + np.random.randint(-100000, 100000)
            ma_5 = price * (1 + np.random.randn() * 0.01)
            rsi_14 = 50 + np.random.randn() * 10
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'close': price,
                'volume': volume,
                'ma_5': ma_5,
                'rsi_14': rsi_14
            })
    
    return pd.DataFrame(data)


def test_grid_search(sample_df):
    """Test grid search."""
    # Define parameter grid
    param_grid = {
        "param1": [0.1, 0.2, 0.3],
        "param2": [5, 10, 15],
        "param3": ["value1", "value2"]
    }
    
    # Split data
    train_df = sample_df[sample_df["timestamp"] < "2023-03-01"]
    val_df = sample_df[sample_df["timestamp"] >= "2023-03-01"]
    
    # Run grid search
    results = grid_search(
        strategy_class=DummyStrategy,
        param_grid=param_grid,
        train_df=train_df,
        val_df=val_df,
        metric="sharpe_ratio",
        maximize=True,
        n_trials=5,
        random_search=True,
        verbose=False
    )
    
    # Check results
    assert "best_params" in results
    assert "best_score" in results
    assert "best_strategy" in results
    assert "results" in results
    
    # Check best params
    assert isinstance(results["best_params"], dict)
    assert "param1" in results["best_params"]
    assert "param2" in results["best_params"]
    assert "param3" in results["best_params"]
    
    # Check best score
    assert isinstance(results["best_score"], float)
    
    # Check best strategy
    assert isinstance(results["best_strategy"], DummyStrategy)
    
    # Check results list
    assert isinstance(results["results"], list)
    assert len(results["results"]) == 5  # n_trials
    assert "params" in results["results"][0]
    assert "score" in results["results"][0]
    assert "metrics" in results["results"][0]


def test_cross_validate(sample_df):
    """Test cross-validation."""
    # Define parameters
    params = {
        "param1": 0.2,
        "param2": 10,
        "param3": "value1"
    }
    
    # Run cross-validation
    results = cross_validate(
        strategy_class=DummyStrategy,
        params=params,
        df=sample_df,
        n_splits=3,
        metric="sharpe_ratio",
        verbose=False
    )
    
    # Check results
    assert "params" in results
    assert "mean_score" in results
    assert "std_score" in results
    assert "scores" in results
    assert "mean_metrics" in results
    assert "metrics" in results
    
    # Check params
    assert results["params"] == params
    
    # Check scores
    assert isinstance(results["mean_score"], float)
    assert isinstance(results["std_score"], float)
    assert isinstance(results["scores"], list)
    assert len(results["scores"]) == 3  # n_splits
    
    # Check metrics
    assert isinstance(results["mean_metrics"], dict)
    assert isinstance(results["metrics"], list)
    assert len(results["metrics"]) == 3  # n_splits


def test_walk_forward_validation(sample_df):
    """Test walk-forward validation."""
    # Define parameters
    params = {
        "param1": 0.2,
        "param2": 10,
        "param3": "value1"
    }
    
    # Run walk-forward validation
    results = walk_forward_validation(
        strategy_class=DummyStrategy,
        params=params,
        df=sample_df,
        train_window=30,
        val_window=10,
        step=10,
        metric="sharpe_ratio",
        verbose=False
    )
    
    # Check results
    assert "params" in results
    assert "mean_score" in results
    assert "std_score" in results
    assert "results" in results
    assert "results_df" in results
    
    # Check params
    assert results["params"] == params
    
    # Check scores
    assert isinstance(results["mean_score"], float)
    assert isinstance(results["std_score"], float)
    
    # Check results
    assert isinstance(results["results"], list)
    assert len(results["results"]) > 0
    assert "window" in results["results"][0]
    assert "train_start" in results["results"][0]
    assert "train_end" in results["results"][0]
    assert "val_start" in results["results"][0]
    assert "val_end" in results["results"][0]
    assert "score" in results["results"][0]
    assert "metrics" in results["results"][0]
    
    # Check results DataFrame
    assert isinstance(results["results_df"], pd.DataFrame)
    assert "window" in results["results_df"].columns
    assert "train_start" in results["results_df"].columns
    assert "train_end" in results["results_df"].columns
    assert "val_start" in results["results_df"].columns
    assert "val_end" in results["results_df"].columns
    assert "score" in results["results_df"].columns
