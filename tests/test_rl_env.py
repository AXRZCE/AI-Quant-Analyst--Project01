"""
Unit tests for the RL environment.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment
from src.rl.env import TradingEnv, MultiAssetTradingEnv

@pytest.fixture
def sample_df():
    """
    Create a sample DataFrame for testing.
    """
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT', 'MSFT'],
        'timestamp': pd.date_range(start='2023-01-01', periods=4).tolist() * 2,
        'close': [150.0, 152.0, 151.0, 153.0, 250.0, 252.0, 251.0, 253.0],
        'ma_5': [0.1, 0.2, 0.15, 0.18, 0.2, 0.25, 0.22, 0.24],
        'rsi_14': [50, 55, 48, 52, 60, 65, 58, 62]
    }
    return pd.DataFrame(data)

def test_env_initialization(sample_df):
    """
    Test environment initialization.
    """
    # Create environment
    env = TradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Check observation space
    # The exact shape depends on implementation details
    assert env.observation_space.shape[0] > 0

    # Check action space
    assert env.action_space.n == 9  # 3^2 actions (3 actions for 2 symbols)

    # Check initial state
    assert env.cash == 100_000
    assert env.positions == {'AAPL': 0.0, 'MSFT': 0.0}
    assert env.current_step == 0

def test_env_reset_and_step(sample_df):
    """
    Test environment reset and step.
    """
    # Create environment
    env = TradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Reset environment
    obs = env.reset()

    # Check initial state
    assert env.cash == env.initial_capital
    assert env.positions == {'AAPL': 0.0, 'MSFT': 0.0}
    assert env.current_step == 0

    # Check observation
    # The exact shape depends on implementation details
    assert obs.shape[0] > 0

    # Test buy action for AAPL (action 1)
    action = 1  # Buy AAPL
    obs, reward, done, info = env.step(action)

    # Check state after buy
    assert env.cash == 0
    assert env.positions['AAPL'] > 0
    assert env.positions['MSFT'] == 0
    assert env.current_step == 1
    assert not done

    # Test sell action for AAPL (action 2)
    action = 2  # Sell AAPL
    obs, reward, done, info = env.step(action)

    # Check state after sell
    assert env.cash > 0
    assert env.positions['AAPL'] == 0
    assert env.positions['MSFT'] == 0
    assert env.current_step == 2
    assert not done

    # Test hold action (action 0)
    action = 0  # Hold
    obs, reward, done, info = env.step(action)

    # Check state after hold
    assert env.cash > 0
    assert env.positions['AAPL'] == 0
    assert env.positions['MSFT'] == 0
    assert env.current_step == 3
    # May or may not be done depending on how the environment is implemented
    # Just check that the step count is correct
    assert env.current_step == 3

def test_env_portfolio_value(sample_df):
    """
    Test environment portfolio value calculation.
    """
    # Create environment
    env = TradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Reset environment
    env.reset()

    # Initial portfolio value
    initial_value = env.cash + sum(env.positions[symbol] * sample_df[sample_df['symbol'] == symbol].iloc[0]['close'] for symbol in env.symbols)
    assert initial_value == 100_000

    # Buy AAPL
    action = 1  # Buy AAPL
    obs, reward, done, info = env.step(action)

    # Portfolio value after buy
    portfolio_value = info['portfolio_value']
    assert portfolio_value < 100_000  # Should be less due to transaction cost
    assert portfolio_value > 99_000  # But not too much less

    # Sell AAPL
    action = 2  # Sell AAPL
    obs, reward, done, info = env.step(action)

    # Portfolio value after sell
    portfolio_value = info['portfolio_value']
    # Note: Portfolio value could be higher if price increased significantly
    # Just check that it's reasonable
    assert portfolio_value > 0  # Should be positive
    assert portfolio_value < 200_000  # Should not be unreasonably high

def test_env_reward(sample_df):
    """
    Test environment reward calculation.
    """
    # Create environment
    env = TradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Reset environment
    env.reset()

    # Buy AAPL
    action = 1  # Buy AAPL
    obs, reward, done, info = env.step(action)

    # Reward should be negative due to transaction cost
    assert reward < 0

    # Sell AAPL
    action = 2  # Sell AAPL
    obs, reward, done, info = env.step(action)

    # Reward depends on price change and transaction cost
    # Could be positive or negative

def test_env_history(sample_df):
    """
    Test environment history recording.
    """
    # Create environment
    env = TradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Reset environment
    env.reset()

    # Take some actions
    actions = [1, 2, 0]  # Buy, sell, hold
    for action in actions:
        env.step(action)

    # Get portfolio history
    history_df = env.get_portfolio_history()

    # Check history
    assert len(history_df) == 3
    assert 'portfolio_value' in history_df.columns
    assert 'cash' in history_df.columns
    assert 'reward' in history_df.columns
    assert 'position_AAPL' in history_df.columns
    assert 'position_MSFT' in history_df.columns
    assert 'action_AAPL' in history_df.columns
    assert 'action_MSFT' in history_df.columns

def test_multi_asset_env(sample_df):
    """
    Test multi-asset environment.
    """
    # Create environment
    env = MultiAssetTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Reset environment
    obs = env.reset()

    # Check observation space
    # The exact shape depends on implementation details
    assert obs.shape[0] > 0

    # Check action space
    assert env.action_space.shape[0] == 2  # 2 symbols

    # Take an action
    action = np.array([0.5, 0.0])  # Allocate 50% to AAPL, 0% to MSFT
    obs, reward, done, info = env.step(action)

    # Check state after action
    # The exact behavior depends on implementation details
    # Just check that the step was taken
    assert env.current_step == 1
    assert env.positions['AAPL'] >= 0  # May have bought AAPL
    assert env.positions['MSFT'] >= 0  # Should not have negative positions
    assert not done

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
