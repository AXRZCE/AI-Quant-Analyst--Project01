"""
Tests for the reinforcement learning strategy wrapper.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl.rl_strategy import RLStrategy, RLStrategyFactory


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
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


def test_rl_strategy_initialization():
    """Test RL strategy initialization."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        price_col="close",
        date_col="timestamp",
        symbol_col="symbol",
        normalize_state=True,
        device="cpu"
    )
    
    # Check attributes
    assert strategy.algorithm == "sac"
    assert strategy.state_dim == 4
    assert strategy.action_dim == 1
    assert strategy.feature_cols == ['close', 'volume', 'ma_5', 'rsi_14']
    assert strategy.price_col == "close"
    assert strategy.date_col == "timestamp"
    assert strategy.symbol_col == "symbol"
    assert strategy.normalize_state == True
    assert strategy.device == "cpu"
    
    # Check agent
    assert strategy.agent is not None


def test_rl_strategy_factory():
    """Test RL strategy factory."""
    # Create strategy
    strategy = RLStrategyFactory.create_strategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        price_col="close",
        date_col="timestamp",
        symbol_col="symbol",
        normalize_state=True,
        device="cpu"
    )
    
    # Check attributes
    assert strategy.algorithm == "sac"
    assert strategy.state_dim == 4
    assert strategy.action_dim == 1
    assert strategy.feature_cols == ['close', 'volume', 'ma_5', 'rsi_14']
    assert strategy.price_col == "close"
    assert strategy.date_col == "timestamp"
    assert strategy.symbol_col == "symbol"
    assert strategy.normalize_state == True
    assert strategy.device == "cpu"
    
    # Check agent
    assert strategy.agent is not None


def test_rl_strategy_fit_normalizer(sample_df):
    """Test RL strategy normalizer fitting."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        normalize_state=True,
        device="cpu"
    )
    
    # Fit normalizer
    strategy.fit_normalizer(sample_df)
    
    # Check normalizer parameters
    assert strategy.state_mean is not None
    assert strategy.state_std is not None
    assert len(strategy.state_mean) == 4
    assert len(strategy.state_std) == 4


def test_rl_strategy_get_state(sample_df):
    """Test RL strategy state extraction."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        normalize_state=False,
        device="cpu"
    )
    
    # Get state
    state = strategy.get_state(sample_df, 0, 'AAPL')
    
    # Check state
    assert state is not None
    assert len(state) == 4
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.float32


def test_rl_strategy_predict(sample_df):
    """Test RL strategy prediction."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        normalize_state=False,
        device="cpu"
    )
    
    # Predict
    actions = strategy.predict(sample_df, 0, ['AAPL', 'MSFT'])
    
    # Check actions
    assert actions is not None
    assert isinstance(actions, dict)
    assert 'AAPL' in actions
    assert 'MSFT' in actions
    assert isinstance(actions['AAPL'], float)
    assert isinstance(actions['MSFT'], float)
    assert -1.0 <= actions['AAPL'] <= 1.0
    assert -1.0 <= actions['MSFT'] <= 1.0


def test_rl_strategy_generate_signals(sample_df):
    """Test RL strategy signal generation."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        normalize_state=False,
        device="cpu"
    )
    
    # Generate signals
    signals_df = strategy.generate_signals(sample_df)
    
    # Check signals
    assert signals_df is not None
    assert isinstance(signals_df, pd.DataFrame)
    assert 'timestamp' in signals_df.columns
    assert 'symbol' in signals_df.columns
    assert 'close' in signals_df.columns
    assert 'signal' in signals_df.columns
    assert 'position' in signals_df.columns
    assert len(signals_df) == len(sample_df['timestamp'].unique()) * len(sample_df['symbol'].unique())
    assert all(-1.0 <= signal <= 1.0 for signal in signals_df['signal'])
    assert all(-1.0 <= position <= 1.0 for position in signals_df['position'])


def test_rl_strategy_backtest(sample_df):
    """Test RL strategy backtesting."""
    # Create strategy
    strategy = RLStrategy(
        algorithm="sac",
        state_dim=4,
        action_dim=1,
        feature_cols=['close', 'volume', 'ma_5', 'rsi_14'],
        normalize_state=False,
        device="cpu"
    )
    
    # Run backtest
    results = strategy.backtest(
        df=sample_df,
        initial_capital=100_000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Check results
    assert results is not None
    assert isinstance(results, dict)
    assert 'initial_capital' in results
    assert 'final_equity' in results
    assert 'total_return' in results
    assert 'annual_return' in results
    assert 'annual_volatility' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'win_rate' in results
    assert 'num_trades' in results
    assert 'equity_curve' in results
    assert 'returns' in results
    assert 'dates' in results
    assert 'trades' in results
    assert results['initial_capital'] == 100_000
    assert len(results['equity_curve']) == len(sample_df['timestamp'].unique()) + 1
    assert len(results['returns']) == len(sample_df['timestamp'].unique()) + 1
    assert len(results['dates']) == len(sample_df['timestamp'].unique()) + 1
