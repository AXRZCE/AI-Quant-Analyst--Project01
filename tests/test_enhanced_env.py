"""
Tests for the enhanced trading environment.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.rl.enhanced_env import EnhancedTradingEnv, EnhancedMultiAssetTradingEnv


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    # Create sample data
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT', 'MSFT'],
        'timestamp': pd.date_range(start='2023-01-01 09:30:00', periods=4, freq='1H').tolist() * 2,
        'close': [150.0, 152.0, 151.0, 153.0, 250.0, 252.0, 251.0, 253.0],
        'volume': [1000000, 1200000, 900000, 1100000, 800000, 850000, 750000, 900000],
        'ma_5': [0.1, 0.2, 0.15, 0.18, 0.2, 0.25, 0.22, 0.24],
        'rsi_14': [50, 55, 48, 52, 60, 65, 58, 62]
    }
    return pd.DataFrame(data)


def test_enhanced_env_initialization(sample_df):
    """Test enhanced environment initialization."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_impact_factor=0.1,
        max_position_size=0.5,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        slippage_std=0.0005,
        market_hours_only=True
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
    assert env.max_position_size == 0.5
    assert env.stop_loss_pct == 0.05
    assert env.take_profit_pct == 0.1
    assert env.market_impact_factor == 0.1
    assert env.slippage_std == 0.0005
    assert env.market_hours_only == True


def test_market_impact(sample_df):
    """Test market impact model."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_impact_factor=0.1
    )

    # Test market impact for buy
    price_before = 150.0
    quantity = 1000
    price_after = env._apply_market_impact('AAPL', price_before, quantity)
    assert price_after > price_before

    # Test market impact for sell
    price_before = 150.0
    quantity = -1000
    price_after = env._apply_market_impact('AAPL', price_before, quantity)
    assert price_after < price_before

    # Test market impact for zero quantity
    price_before = 150.0
    quantity = 0
    price_after = env._apply_market_impact('AAPL', price_before, quantity)
    assert price_after == price_before


def test_slippage(sample_df):
    """Test slippage model."""
    # Create environment with fixed seed for reproducibility
    np.random.seed(42)
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        slippage_std=0.01
    )

    # Test slippage
    price_before = 150.0
    price_after = env._apply_slippage(price_before)
    assert price_after != price_before

    # Test slippage with zero std
    env.slippage_std = 0.0
    price_before = 150.0
    price_after = env._apply_slippage(price_before)
    assert price_after == price_before


def test_market_hours(sample_df):
    """Test market hours constraints."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_hours_only=True,
        market_open_time="09:00",
        market_close_time="16:00"
    )

    # Test within market hours
    assert env._check_market_hours("2023-01-01 10:30:00")
    assert env._check_market_hours("2023-01-01 15:59:00")

    # Test outside market hours
    assert not env._check_market_hours("2023-01-01 08:59:00")
    assert not env._check_market_hours("2023-01-01 16:01:00")


def test_risk_limits(sample_df):
    """Test risk management limits."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        max_drawdown_pct=0.2
    )

    # Set up a position
    env.positions['AAPL'] = 100
    env.entry_prices['AAPL'] = 150.0

    # Test stop loss
    assert env._check_risk_limits('AAPL', 142.0)  # Price below stop loss
    assert not env._check_risk_limits('AAPL', 143.0)  # Price above stop loss

    # Test take profit
    assert env._check_risk_limits('AAPL', 165.0)  # Price above take profit
    assert not env._check_risk_limits('AAPL', 164.0)  # Price below take profit

    # Test max drawdown
    env.max_portfolio_value = 120_000
    env.cash = 0
    env.positions['AAPL'] = 100
    env.positions['MSFT'] = 100
    # Current portfolio value: 100*150 + 100*250 = 40,000
    # Drawdown: (120,000 - 40,000) / 120,000 = 0.67
    assert env._check_risk_limits('AAPL', 150.0)  # Drawdown exceeds limit


def test_position_sizing(sample_df):
    """Test position sizing constraints."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        max_position_size=0.5
    )

    # Reset environment
    obs, _ = env.reset()

    # Take buy action
    action = 1  # Buy AAPL
    obs, reward, terminated, truncated, info = env.step(action)

    # Check position size
    # With max_position_size=0.5, should only use 50% of capital
    assert env.cash < 100_000  # Some cash was used
    assert env.cash > 0  # Not all cash was used
    assert env.positions['AAPL'] > 0  # Position was opened

    # Calculate expected position
    expected_max_capital = 100_000 * 0.5
    expected_position = expected_max_capital / 150.0 * (1 - 0.001)  # Apply transaction cost
    assert abs(env.positions['AAPL'] - expected_position) < 1.0  # Allow for small rounding differences


def test_enhanced_env_step(sample_df):
    """Test enhanced environment step function."""
    # Create environment
    env = EnhancedTradingEnv(
        df=sample_df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_impact_factor=0.1,
        max_position_size=0.5,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        slippage_std=0.0005
    )

    # Reset environment
    obs, _ = env.reset()

    # Take some actions
    actions = [1, 0, 2]  # Buy, hold, sell
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)

        # Check observation
        assert obs is not None
        assert len(obs) == env.observation_space.shape[0]

        # Check info
        assert "portfolio_value" in info
        assert "cash" in info
        assert "positions" in info
        assert "drawdown_pct" in info
        assert "max_portfolio_value" in info

    # Check history
    history = env.history
    assert len(history) == 3
    assert "drawdown_pct" in history[0]
    assert "portfolio_value" in history[0]
    assert "cash" in history[0]
    assert "positions" in history[0]
    assert "reward" in history[0]


@pytest.fixture
def sample_df_with_sector():
    """Create a sample DataFrame with sector information for testing."""
    # Create sample data
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL'],
        'timestamp': pd.date_range(start='2023-01-01 09:30:00', periods=4, freq='1H').tolist() * 3,
        'close': [150.0, 152.0, 151.0, 153.0, 250.0, 252.0, 251.0, 253.0, 2000.0, 2020.0, 2010.0, 2030.0],
        'volume': [1000000, 1200000, 900000, 1100000, 800000, 850000, 750000, 900000, 500000, 550000, 480000, 520000],
        'ma_5': [0.1, 0.2, 0.15, 0.18, 0.2, 0.25, 0.22, 0.24, 0.3, 0.32, 0.29, 0.33],
        'rsi_14': [50, 55, 48, 52, 60, 65, 58, 62, 45, 48, 43, 47],
        'sector': ['Technology', 'Technology', 'Technology', 'Technology',
                  'Technology', 'Technology', 'Technology', 'Technology',
                  'Communication', 'Communication', 'Communication', 'Communication']
    }
    return pd.DataFrame(data)


def test_enhanced_multi_asset_env_initialization(sample_df_with_sector):
    """Test enhanced multi-asset environment initialization."""
    # Create environment
    env = EnhancedMultiAssetTradingEnv(
        df=sample_df_with_sector,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_impact_factor=0.1,
        max_position_size=0.5,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        slippage_std=0.0005,
        market_hours_only=True,
        max_sector_exposure={'Technology': 0.6, 'Communication': 0.4},
        max_concentration=0.3,
        sector_col='sector'
    )

    # Check observation space
    assert env.observation_space.shape[0] > 0

    # Check action space
    assert env.action_space.shape[0] == 3  # 3 symbols

    # Check initial state
    assert env.cash == 100_000
    assert env.positions == {'AAPL': 0.0, 'MSFT': 0.0, 'GOOGL': 0.0}
    assert env.current_step == 0
    assert env.max_position_size == 0.5
    assert env.stop_loss_pct == 0.05
    assert env.take_profit_pct == 0.1
    assert env.market_impact_factor == 0.1
    assert env.slippage_std == 0.0005
    assert env.market_hours_only == True
    assert env.max_sector_exposure == {'Technology': 0.6, 'Communication': 0.4}
    assert env.max_concentration == 0.3
    assert env.symbol_to_sector == {'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication'}


def test_sector_exposure(sample_df_with_sector):
    """Test sector exposure calculation and constraints."""
    # Create environment
    env = EnhancedMultiAssetTradingEnv(
        df=sample_df_with_sector,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        max_sector_exposure={'Technology': 0.6, 'Communication': 0.4},
        max_concentration=0.3,
        sector_col='sector'
    )

    # Reset environment
    obs, _ = env.reset()

    # Set up positions
    prices = {'AAPL': 150.0, 'MSFT': 250.0, 'GOOGL': 2000.0}
    env.positions = {'AAPL': 200, 'MSFT': 100, 'GOOGL': 10}  # 30k + 25k + 20k = 75k
    env.cash = 25000  # Total portfolio value: 100k

    # Calculate sector exposure
    sector_exposure = env._get_sector_exposure(prices)

    # Check sector exposure
    assert abs(sector_exposure['Technology'] - 0.55) < 0.01  # (30k + 25k) / 100k = 0.55
    assert abs(sector_exposure['Communication'] - 0.20) < 0.01  # 20k / 100k = 0.20

    # Test portfolio constraints
    # Try to increase Technology exposure beyond limit
    target_allocation = 0.7  # Would make Technology exposure 0.7 + 0.25 = 0.95
    adjusted_allocation = env._check_portfolio_constraints('AAPL', target_allocation, prices)
    assert adjusted_allocation < target_allocation  # Should be reduced

    # Try to increase Communication exposure within limit
    target_allocation = 0.3  # Would make Communication exposure 0.3 + 0.0 = 0.3
    adjusted_allocation = env._check_portfolio_constraints('GOOGL', target_allocation, prices)
    assert adjusted_allocation == target_allocation  # Should not be reduced


def test_enhanced_multi_asset_env_step(sample_df_with_sector):
    """Test enhanced multi-asset environment step function."""
    # Create environment
    env = EnhancedMultiAssetTradingEnv(
        df=sample_df_with_sector,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001,
        market_impact_factor=0.1,
        max_position_size=0.5,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        slippage_std=0.0005,
        max_sector_exposure={'Technology': 0.6, 'Communication': 0.4},
        max_concentration=0.3,
        sector_col='sector'
    )

    # Reset environment
    obs, _ = env.reset()

    # Take some actions
    actions = [
        np.array([0.3, 0.2, 0.1]),  # Allocate to all assets
        np.array([0.0, 0.0, 0.0]),    # Hold
        np.array([-0.3, -0.2, -0.1])   # Reduce positions
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)

        # Check observation
        assert obs is not None
        assert len(obs) == env.observation_space.shape[0]

        # Check info
        assert "portfolio_value" in info
        assert "cash" in info
        assert "positions" in info
        assert "drawdown_pct" in info
        assert "max_portfolio_value" in info
        assert "sector_exposure" in info

    # Check history
    history = env.history
    assert len(history) == 3
    assert "drawdown_pct" in history[0]
    assert "portfolio_value" in history[0]
    assert "cash" in history[0]
    assert "positions" in history[0]
    assert "reward" in history[0]
    assert "sector_exposure" in history[0]
