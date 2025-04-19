"""
Tests for the reinforcement learning evaluation framework.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl.evaluation import (
    calculate_performance_metrics,
    evaluate_strategy,
    compare_strategies
)
from src.rl.rl_strategy import RLStrategy


class DummyStrategy:
    """Dummy strategy for testing."""
    
    def __init__(self, return_pattern="random"):
        """Initialize dummy strategy."""
        self.return_pattern = return_pattern
    
    def backtest(self, df, initial_capital=100_000, transaction_cost=0.001, slippage=0.0005):
        """Run dummy backtest."""
        # Get unique dates
        dates = df["timestamp"].unique()
        
        # Generate dummy equity curve and returns
        equity_curve = [initial_capital]
        returns = [0.0]
        
        for i in range(len(dates)):
            if self.return_pattern == "random":
                # Random returns between -1% and 1%
                ret = np.random.uniform(-0.01, 0.01)
            elif self.return_pattern == "positive":
                # Positive returns between 0% and 1%
                ret = np.random.uniform(0.0, 0.01)
            elif self.return_pattern == "negative":
                # Negative returns between -1% and 0%
                ret = np.random.uniform(-0.01, 0.0)
            elif self.return_pattern == "trend":
                # Trending returns (first half positive, second half negative)
                if i < len(dates) // 2:
                    ret = np.random.uniform(0.0, 0.01)
                else:
                    ret = np.random.uniform(-0.01, 0.0)
            else:
                ret = 0.0
            
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


def test_calculate_performance_metrics():
    """Test performance metrics calculation."""
    # Create dummy equity curve and returns
    equity_curve = [100_000]
    returns = [0.0]
    
    # Generate random returns
    for _ in range(252):  # One year of daily returns
        ret = np.random.normal(0.0005, 0.01)  # Mean 0.05% daily return, 1% daily vol
        equity_curve.append(equity_curve[-1] * (1 + ret))
        returns.append(ret)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(equity_curve, returns)
    
    # Check metrics
    assert "total_return" in metrics
    assert "annual_return" in metrics
    assert "annual_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "calmar_ratio" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    
    # Check metric values
    assert isinstance(metrics["total_return"], float)
    assert isinstance(metrics["annual_return"], float)
    assert isinstance(metrics["annual_volatility"], float)
    assert isinstance(metrics["sharpe_ratio"], float)
    assert isinstance(metrics["sortino_ratio"], float)
    assert isinstance(metrics["max_drawdown"], float)
    assert isinstance(metrics["calmar_ratio"], float)
    assert isinstance(metrics["win_rate"], float)
    assert isinstance(metrics["profit_factor"], float)
    
    # Check metric ranges
    assert -1.0 <= metrics["max_drawdown"] <= 0.0
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["profit_factor"] >= 0.0


def test_evaluate_strategy(sample_df):
    """Test strategy evaluation."""
    # Create dummy strategy
    strategy = DummyStrategy(return_pattern="random")
    
    # Evaluate strategy
    results = evaluate_strategy(
        strategy=strategy,
        test_df=sample_df,
        initial_capital=100_000,
        transaction_cost=0.001,
        slippage=0.0005,
        risk_free_rate=0.0
    )
    
    # Check results
    assert "backtest_results" in results
    assert "metrics" in results
    
    # Check backtest results
    assert "initial_capital" in results["backtest_results"]
    assert "final_equity" in results["backtest_results"]
    assert "total_return" in results["backtest_results"]
    assert "equity_curve" in results["backtest_results"]
    assert "returns" in results["backtest_results"]
    assert "dates" in results["backtest_results"]
    assert "trades" in results["backtest_results"]
    
    # Check metrics
    assert "total_return" in results["metrics"]
    assert "annual_return" in results["metrics"]
    assert "annual_volatility" in results["metrics"]
    assert "sharpe_ratio" in results["metrics"]
    assert "sortino_ratio" in results["metrics"]
    assert "max_drawdown" in results["metrics"]
    assert "calmar_ratio" in results["metrics"]
    assert "win_rate" in results["metrics"]
    assert "profit_factor" in results["metrics"]


def test_compare_strategies(sample_df):
    """Test strategy comparison."""
    # Create dummy strategies
    strategies = {
        "Random": DummyStrategy(return_pattern="random"),
        "Positive": DummyStrategy(return_pattern="positive"),
        "Negative": DummyStrategy(return_pattern="negative"),
        "Trend": DummyStrategy(return_pattern="trend")
    }
    
    # Compare strategies
    results = compare_strategies(
        strategies=strategies,
        test_df=sample_df,
        initial_capital=100_000,
        transaction_cost=0.001,
        slippage=0.0005,
        risk_free_rate=0.0
    )
    
    # Check results
    assert "Random" in results
    assert "Positive" in results
    assert "Negative" in results
    assert "Trend" in results
    assert "comparison" in results
    
    # Check comparison DataFrame
    assert isinstance(results["comparison"], pd.DataFrame)
    assert results["comparison"].shape == (4, 9)  # 4 strategies, 9 metrics
    assert list(results["comparison"].index) == ["Random", "Positive", "Negative", "Trend"]
    assert "total_return" in results["comparison"].columns
    assert "annual_return" in results["comparison"].columns
    assert "annual_volatility" in results["comparison"].columns
    assert "sharpe_ratio" in results["comparison"].columns
    assert "sortino_ratio" in results["comparison"].columns
    assert "max_drawdown" in results["comparison"].columns
    assert "calmar_ratio" in results["comparison"].columns
    assert "win_rate" in results["comparison"].columns
    assert "profit_factor" in results["comparison"].columns
