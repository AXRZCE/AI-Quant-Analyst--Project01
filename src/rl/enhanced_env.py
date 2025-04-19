"""
Enhanced reinforcement learning environment for trading with realistic constraints.

This module provides an enhanced trading environment with realistic constraints
such as market impact, position sizing limits, risk management, and more.
"""

import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import base environment
from src.rl.env import TradingEnv, MultiAssetTradingEnv


class EnhancedTradingEnv(TradingEnv):
    """
    Enhanced trading environment with realistic constraints.

    Adds the following features to the base TradingEnv:
    - Market impact model
    - Position sizing constraints
    - Risk management (stop-loss, take-profit)
    - Slippage model
    - Market hours constraints
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 1,
        symbol_col: str = "symbol",
        price_col: str = "close",
        date_col: Optional[str] = "timestamp",
        # Enhanced parameters
        market_impact_factor: float = 0.1,
        max_position_size: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        slippage_std: float = 0.0005,
        market_hours_only: bool = False,
        market_open_time: str = "09:30",
        market_close_time: str = "16:00",
        max_drawdown_pct: Optional[float] = None
    ):
        """
        Initialize the enhanced trading environment.

        Args:
            df: DataFrame with price and feature data
            feature_cols: List of feature column names
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction of trade value
            reward_scaling: Scaling factor for rewards
            window_size: Number of time steps to include in the observation
            symbol_col: Name of the symbol column
            price_col: Name of the price column
            date_col: Name of the date column
            market_impact_factor: Factor for market impact model
            max_position_size: Maximum position size as a fraction of capital
            stop_loss_pct: Stop loss percentage (None to disable)
            take_profit_pct: Take profit percentage (None to disable)
            slippage_std: Standard deviation for slippage model
            market_hours_only: Whether to only trade during market hours
            market_open_time: Market open time (HH:MM)
            market_close_time: Market close time (HH:MM)
            max_drawdown_pct: Maximum allowed drawdown percentage (None to disable)
        """
        # Initialize parent class
        super().__init__(
            df=df,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            reward_scaling=reward_scaling,
            window_size=window_size,
            symbol_col=symbol_col,
            price_col=price_col,
            date_col=date_col
        )

        # Store enhanced parameters
        self.market_impact_factor = market_impact_factor
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage_std = slippage_std
        self.market_hours_only = market_hours_only
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.max_drawdown_pct = max_drawdown_pct

        # Initialize additional state variables
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.max_portfolio_value = initial_capital
        self.current_drawdown_pct = 0.0

        # Expand observation space to include additional features
        # Add entry price ratio, drawdown, and days held for each symbol
        obs_dim = self.observation_space.shape[0] + 3 * self.n_symbols
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize days held counter
        self.days_held = {symbol: 0 for symbol in self.symbols}

    def _apply_market_impact(self, symbol: str, price: float, quantity: float) -> float:
        """
        Apply market impact model to adjust execution price.

        Args:
            symbol: Symbol being traded
            price: Current market price
            quantity: Quantity being traded (positive for buy, negative for sell)

        Returns:
            Adjusted price after market impact
        """
        # Simple square-root market impact model
        # Price impact is proportional to the square root of the quantity
        # and scaled by the market impact factor
        if quantity == 0:
            return price

        # Get average daily volume if available
        avg_volume = 1000000  # Default value
        if 'volume' in self.df.columns:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if not symbol_df.empty and 'volume' in symbol_df.columns:
                avg_volume = symbol_df['volume'].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    avg_volume = 1000000

        # Calculate market impact
        impact = self.market_impact_factor * price * (abs(quantity) / avg_volume) ** 0.5

        # Adjust price based on direction
        if quantity > 0:  # Buy
            return price * (1 + impact)
        else:  # Sell
            return price * (1 - impact)

    def _apply_slippage(self, price: float) -> float:
        """
        Apply slippage model to adjust execution price.

        Args:
            price: Current market price

        Returns:
            Adjusted price after slippage
        """
        # Simple normal distribution slippage model
        if self.slippage_std <= 0:
            return price

        # Generate random slippage
        slippage = np.random.normal(0, self.slippage_std)

        # Adjust price
        return price * (1 + slippage)

    def _check_market_hours(self, timestamp) -> bool:
        """
        Check if current timestamp is within market hours.

        Args:
            timestamp: Current timestamp

        Returns:
            True if within market hours, False otherwise
        """
        if not self.market_hours_only or self.date_col is None:
            return True

        # Convert timestamp to time
        if isinstance(timestamp, str):
            try:
                time_str = timestamp.split(' ')[1][:5]  # Extract HH:MM
            except (IndexError, AttributeError):
                return True
        elif hasattr(timestamp, 'time'):
            time_str = timestamp.time().strftime('%H:%M')
        else:
            return True

        # Check if within market hours
        return self.market_open_time <= time_str <= self.market_close_time

    def _check_risk_limits(self, symbol: str, price: float) -> bool:
        """
        Check if position should be closed due to risk limits.

        Args:
            symbol: Symbol to check
            price: Current price

        Returns:
            True if position should be closed, False otherwise
        """
        # Check if position exists
        if self.positions[symbol] <= 0:
            return False

        # Check stop loss
        if self.stop_loss_pct is not None and self.entry_prices[symbol] > 0:
            stop_price = self.entry_prices[symbol] * (1 - self.stop_loss_pct)
            if price <= stop_price:
                logger.info(f"Stop loss triggered for {symbol} at {price:.2f}")
                return True

        # Check take profit
        if self.take_profit_pct is not None and self.entry_prices[symbol] > 0:
            take_profit_price = self.entry_prices[symbol] * (1 + self.take_profit_pct)
            if price >= take_profit_price:
                logger.info(f"Take profit triggered for {symbol} at {price:.2f}")
                return True

        # Check max drawdown
        if self.max_drawdown_pct is not None:
            portfolio_value = self.cash + sum(self.positions[s] * self._get_price(s) for s in self.symbols)
            drawdown_pct = 1 - portfolio_value / self.max_portfolio_value
            self.current_drawdown_pct = max(0, drawdown_pct)

            if drawdown_pct >= self.max_drawdown_pct:
                logger.info(f"Max drawdown triggered at {drawdown_pct:.2%}")
                return True

        return False

    def _get_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price
        """
        symbol_df = self.df[self.df[self.symbol_col] == symbol]
        if self.current_step < len(symbol_df):
            return symbol_df.iloc[self.current_step][self.price_col]
        else:
            # If we've reached the end of data for this symbol, use the last price
            return symbol_df.iloc[-1][self.price_col]

    def _get_timestamp(self, symbol: str) -> Any:
        """
        Get current timestamp for a symbol.

        Args:
            symbol: Symbol to get timestamp for

        Returns:
            Current timestamp
        """
        if self.date_col is None:
            return None

        symbol_df = self.df[self.df[self.symbol_col] == symbol]
        if self.current_step < len(symbol_df):
            return symbol_df.iloc[self.current_step][self.date_col]
        else:
            # If we've reached the end of data for this symbol, use the last timestamp
            return symbol_df.iloc[-1][self.date_col]

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            Current observation
        """
        # Get base observation from parent class
        base_obs = super()._get_obs()

        # Add enhanced features
        enhanced_features = []

        for symbol in self.symbols:
            # Add entry price ratio
            price = self._get_price(symbol)
            entry_price_ratio = self.entry_prices[symbol] / price if price > 0 and self.entry_prices[symbol] > 0 else 0
            enhanced_features.append(entry_price_ratio)

            # Add days held
            days_held_normalized = min(self.days_held[symbol] / 30, 1.0)  # Normalize to [0, 1]
            enhanced_features.append(days_held_normalized)

            # Add drawdown
            enhanced_features.append(self.current_drawdown_pct)

        # Combine base observation and enhanced features
        obs = np.concatenate([base_obs, enhanced_features]).astype(np.float32)

        return obs

    def step(self, action: int):
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current prices and timestamps
        prices = {}
        timestamps = {}
        for symbol in self.symbols:
            prices[symbol] = self._get_price(symbol)
            timestamps[symbol] = self._get_timestamp(symbol)

        # Calculate portfolio value before action
        portfolio_value_before = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)

        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value_before)

        # Decode action
        symbol_actions = self._decode_action(action)

        # Execute actions
        for symbol, symbol_action in symbol_actions.items():
            price = prices[symbol]
            timestamp = timestamps[symbol]

            # Check if within market hours
            if not self._check_market_hours(timestamp):
                continue

            # Check risk limits
            if self._check_risk_limits(symbol, price):
                # Close position
                symbol_action = 2

            # Apply slippage to price
            execution_price = self._apply_slippage(price)

            if symbol_action == 1 and self.cash > 0:  # Buy
                # Calculate position size (limited by max_position_size)
                max_capital = self.initial_capital * self.max_position_size
                position_size = min(self.cash, max_capital) / execution_price

                # Apply market impact
                execution_price = self._apply_market_impact(symbol, execution_price, position_size)

                # Recalculate position size with updated price
                position_size = min(self.cash, max_capital) / execution_price

                # Apply transaction cost
                position_size *= (1 - self.transaction_cost)

                # Update positions and cash
                self.positions[symbol] += position_size
                self.cash = 0

                # Record entry price
                self.entry_prices[symbol] = execution_price

                # Reset days held
                self.days_held[symbol] = 0

            elif symbol_action == 2 and self.positions[symbol] > 0:  # Sell
                # Calculate sale value
                position_size = self.positions[symbol]

                # Apply market impact
                execution_price = self._apply_market_impact(symbol, execution_price, -position_size)

                # Calculate sale value
                sale_value = position_size * execution_price

                # Apply transaction cost
                sale_value *= (1 - self.transaction_cost)

                # Update positions and cash
                self.cash += sale_value
                self.positions[symbol] = 0

                # Reset entry price
                self.entry_prices[symbol] = 0

                # Reset days held
                self.days_held[symbol] = 0

            else:  # Hold
                # Increment days held
                self.days_held[symbol] += 1

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = True
        truncated = False
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                terminated = False
                break

        # Calculate portfolio value after action
        portfolio_value_after = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)

        # Calculate reward (change in portfolio value)
        reward = (portfolio_value_after - portfolio_value_before) * self.reward_scaling

        # Get next observation
        obs = self._get_obs()

        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "symbol_actions": symbol_actions,
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "reward": reward,
            "drawdown_pct": self.current_drawdown_pct
        })

        # Create info dictionary
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "step": self.current_step,
            "drawdown_pct": self.current_drawdown_pct,
            "max_portfolio_value": self.max_portfolio_value
        }

        return obs, reward, terminated, truncated, info


class EnhancedMultiAssetTradingEnv(MultiAssetTradingEnv):
    """
    Enhanced multi-asset trading environment with realistic constraints.

    Adds the following features to the base MultiAssetTradingEnv:
    - Market impact model
    - Position sizing constraints
    - Risk management (stop-loss, take-profit)
    - Slippage model
    - Market hours constraints
    - Portfolio constraints (sector exposure, concentration limits)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 1,
        symbol_col: str = "symbol",
        price_col: str = "close",
        date_col: Optional[str] = "timestamp",
        max_position_size: float = 1.0,
        # Enhanced parameters
        market_impact_factor: float = 0.1,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        slippage_std: float = 0.0005,
        market_hours_only: bool = False,
        market_open_time: str = "09:30",
        market_close_time: str = "16:00",
        max_drawdown_pct: Optional[float] = None,
        max_sector_exposure: Optional[Dict[str, float]] = None,
        max_concentration: float = 0.3,
        sector_col: Optional[str] = None
    ):
        """
        Initialize the enhanced multi-asset trading environment.

        Args:
            df: DataFrame with price and feature data
            feature_cols: List of feature column names
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction of trade value
            reward_scaling: Scaling factor for rewards
            window_size: Number of time steps to include in the observation
            symbol_col: Name of the symbol column
            price_col: Name of the price column
            date_col: Name of the date column
            max_position_size: Maximum position size as a fraction of capital
            market_impact_factor: Factor for market impact model
            stop_loss_pct: Stop loss percentage (None to disable)
            take_profit_pct: Take profit percentage (None to disable)
            slippage_std: Standard deviation for slippage model
            market_hours_only: Whether to only trade during market hours
            market_open_time: Market open time (HH:MM)
            market_close_time: Market close time (HH:MM)
            max_drawdown_pct: Maximum allowed drawdown percentage (None to disable)
            max_sector_exposure: Maximum exposure per sector (None to disable)
            max_concentration: Maximum concentration in a single asset
            sector_col: Name of the sector column
        """
        # Initialize parent class
        super().__init__(
            df=df,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            reward_scaling=reward_scaling,
            window_size=window_size,
            symbol_col=symbol_col,
            price_col=price_col,
            date_col=date_col,
            max_position_size=max_position_size
        )

        # Store enhanced parameters
        self.market_impact_factor = market_impact_factor
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage_std = slippage_std
        self.market_hours_only = market_hours_only
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.max_drawdown_pct = max_drawdown_pct
        self.max_sector_exposure = max_sector_exposure or {}
        self.max_concentration = max_concentration
        self.sector_col = sector_col

        # Initialize additional state variables
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.max_portfolio_value = initial_capital
        self.current_drawdown_pct = 0.0

        # Get sector mapping if sector column is provided
        self.symbol_to_sector = {}
        if self.sector_col and self.sector_col in self.df.columns:
            for symbol in self.symbols:
                symbol_df = self.df[self.df[self.symbol_col] == symbol]
                if not symbol_df.empty and self.sector_col in symbol_df.columns:
                    sector = symbol_df[self.sector_col].iloc[0]
                    self.symbol_to_sector[symbol] = sector

        # Expand observation space to include additional features
        # Add entry price ratio, drawdown, and days held for each symbol
        obs_dim = self.observation_space.shape[0] + 3 * self.n_symbols
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize days held counter
        self.days_held = {symbol: 0 for symbol in self.symbols}

    def _apply_market_impact(self, symbol: str, price: float, quantity: float) -> float:
        """
        Apply market impact model to adjust execution price.

        Args:
            symbol: Symbol being traded
            price: Current market price
            quantity: Quantity being traded (positive for buy, negative for sell)

        Returns:
            Adjusted price after market impact
        """
        # Simple square-root market impact model
        # Price impact is proportional to the square root of the quantity
        # and scaled by the market impact factor
        if quantity == 0:
            return price

        # Get average daily volume if available
        avg_volume = 1000000  # Default value
        if 'volume' in self.df.columns:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if not symbol_df.empty and 'volume' in symbol_df.columns:
                avg_volume = symbol_df['volume'].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    avg_volume = 1000000

        # Calculate market impact
        impact = self.market_impact_factor * price * (abs(quantity) / avg_volume) ** 0.5

        # Adjust price based on direction
        if quantity > 0:  # Buy
            return price * (1 + impact)
        else:  # Sell
            return price * (1 - impact)

    def _apply_slippage(self, price: float) -> float:
        """
        Apply slippage model to adjust execution price.

        Args:
            price: Current market price

        Returns:
            Adjusted price after slippage
        """
        # Simple normal distribution slippage model
        if self.slippage_std <= 0:
            return price

        # Generate random slippage
        slippage = np.random.normal(0, self.slippage_std)

        # Adjust price
        return price * (1 + slippage)

    def _check_market_hours(self, timestamp) -> bool:
        """
        Check if current timestamp is within market hours.

        Args:
            timestamp: Current timestamp

        Returns:
            True if within market hours, False otherwise
        """
        if not self.market_hours_only or self.date_col is None:
            return True

        # Convert timestamp to time
        if isinstance(timestamp, str):
            try:
                time_str = timestamp.split(' ')[1][:5]  # Extract HH:MM
            except (IndexError, AttributeError):
                return True
        elif hasattr(timestamp, 'time'):
            time_str = timestamp.time().strftime('%H:%M')
        else:
            return True

        # Check if within market hours
        return self.market_open_time <= time_str <= self.market_close_time

    def _check_risk_limits(self, symbol: str, price: float) -> bool:
        """
        Check if position should be closed due to risk limits.

        Args:
            symbol: Symbol to check
            price: Current price

        Returns:
            True if position should be closed, False otherwise
        """
        # Check if position exists
        if self.positions[symbol] <= 0:
            return False

        # Check stop loss
        if self.stop_loss_pct is not None and self.entry_prices[symbol] > 0:
            stop_price = self.entry_prices[symbol] * (1 - self.stop_loss_pct)
            if price <= stop_price:
                logger.info(f"Stop loss triggered for {symbol} at {price:.2f}")
                return True

        # Check take profit
        if self.take_profit_pct is not None and self.entry_prices[symbol] > 0:
            take_profit_price = self.entry_prices[symbol] * (1 + self.take_profit_pct)
            if price >= take_profit_price:
                logger.info(f"Take profit triggered for {symbol} at {price:.2f}")
                return True

        # Check max drawdown
        if self.max_drawdown_pct is not None:
            portfolio_value = self.cash + sum(self.positions[s] * self._get_price(s) for s in self.symbols)
            drawdown_pct = 1 - portfolio_value / self.max_portfolio_value
            self.current_drawdown_pct = max(0, drawdown_pct)

            if drawdown_pct >= self.max_drawdown_pct:
                logger.info(f"Max drawdown triggered at {drawdown_pct:.2%}")
                return True

        return False

    def _get_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price
        """
        symbol_df = self.df[self.df[self.symbol_col] == symbol]
        if self.current_step < len(symbol_df):
            return symbol_df.iloc[self.current_step][self.price_col]
        else:
            # If we've reached the end of data for this symbol, use the last price
            return symbol_df.iloc[-1][self.price_col]

    def _get_timestamp(self, symbol: str) -> Any:
        """
        Get current timestamp for a symbol.

        Args:
            symbol: Symbol to get timestamp for

        Returns:
            Current timestamp
        """
        if self.date_col is None:
            return None

        symbol_df = self.df[self.df[self.symbol_col] == symbol]
        if self.current_step < len(symbol_df):
            return symbol_df.iloc[self.current_step][self.date_col]
        else:
            # If we've reached the end of data for this symbol, use the last timestamp
            return symbol_df.iloc[-1][self.date_col]

    def _get_sector_exposure(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current sector exposure.

        Args:
            prices: Dictionary of current prices

        Returns:
            Dictionary of sector exposures as fractions of portfolio value
        """
        if not self.symbol_to_sector:
            return {}

        # Calculate portfolio value
        portfolio_value = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)

        # Calculate sector exposure
        sector_exposure = {}
        for symbol, sector in self.symbol_to_sector.items():
            position_value = self.positions[symbol] * prices[symbol]
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value / portfolio_value if portfolio_value > 0 else 0

        return sector_exposure

    def _check_portfolio_constraints(self, symbol: str, target_allocation: float, prices: Dict[str, float]) -> float:
        """
        Check portfolio constraints and adjust target allocation if needed.

        Args:
            symbol: Symbol to check
            target_allocation: Target allocation for the symbol
            prices: Dictionary of current prices

        Returns:
            Adjusted target allocation
        """
        # Calculate portfolio value
        portfolio_value = self.cash + sum(self.positions[s] * prices[s] for s in self.symbols)

        # Check concentration limit
        if target_allocation > self.max_concentration:
            logger.info(f"Concentration limit exceeded for {symbol}, adjusting allocation from {target_allocation:.2f} to {self.max_concentration:.2f}")
            target_allocation = self.max_concentration

        # Check sector exposure limit
        if self.symbol_to_sector and symbol in self.symbol_to_sector:
            sector = self.symbol_to_sector[symbol]
            if sector in self.max_sector_exposure:
                # Calculate current sector exposure
                sector_exposure = self._get_sector_exposure(prices)
                current_exposure = sector_exposure.get(sector, 0)

                # Calculate current allocation for this symbol
                current_allocation = self.positions[symbol] * prices[symbol] / portfolio_value if portfolio_value > 0 else 0

                # Calculate maximum additional allocation
                max_additional = self.max_sector_exposure[sector] - current_exposure + current_allocation

                if target_allocation > max_additional:
                    logger.info(f"Sector exposure limit exceeded for {sector}, adjusting allocation from {target_allocation:.2f} to {max_additional:.2f}")
                    target_allocation = max(0, max_additional)

        return target_allocation

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            Current observation
        """
        # Get base observation from parent class
        base_obs = super()._get_obs()

        # Add enhanced features
        enhanced_features = []

        for symbol in self.symbols:
            # Add entry price ratio
            price = self._get_price(symbol)
            entry_price_ratio = self.entry_prices[symbol] / price if price > 0 and self.entry_prices[symbol] > 0 else 0
            enhanced_features.append(entry_price_ratio)

            # Add days held
            days_held_normalized = min(self.days_held[symbol] / 30, 1.0)  # Normalize to [0, 1]
            enhanced_features.append(days_held_normalized)

            # Add drawdown
            enhanced_features.append(self.current_drawdown_pct)

        # Combine base observation and enhanced features
        obs = np.concatenate([base_obs, enhanced_features]).astype(np.float32)

        return obs

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.

        Args:
            action: Action to take (allocation of capital across assets)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current prices and timestamps
        prices = {}
        timestamps = {}
        for symbol in self.symbols:
            prices[symbol] = self._get_price(symbol)
            timestamps[symbol] = self._get_timestamp(symbol)

        # Calculate portfolio value before action
        portfolio_value_before = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)

        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value_before)

        # Normalize action to sum to 1
        action = np.clip(action, -1.0, 1.0)

        # Calculate target allocation
        target_allocation = {}
        for i, symbol in enumerate(self.symbols):
            target_allocation[symbol] = action[i]

            # Check risk limits
            if self._check_risk_limits(symbol, prices[symbol]):
                # Close position
                target_allocation[symbol] = -1.0

            # Check portfolio constraints
            target_allocation[symbol] = self._check_portfolio_constraints(symbol, target_allocation[symbol], prices)

        # Calculate current allocation
        current_allocation = {}
        for symbol in self.symbols:
            current_allocation[symbol] = self.positions[symbol] * prices[symbol] / portfolio_value_before if portfolio_value_before > 0 else 0

        # Calculate trades
        trades = {}
        for symbol in self.symbols:
            # Check if within market hours
            if not self._check_market_hours(timestamps[symbol]):
                continue

            # Calculate trade value
            trade_value = (target_allocation[symbol] - current_allocation[symbol]) * portfolio_value_before

            # Apply slippage to price
            execution_price = self._apply_slippage(prices[symbol])

            # Apply market impact
            if trade_value > 0:  # Buy
                position_size = trade_value / execution_price
                execution_price = self._apply_market_impact(symbol, execution_price, position_size)
                trade_value = position_size * execution_price
            elif trade_value < 0:  # Sell
                position_size = -trade_value / execution_price
                execution_price = self._apply_market_impact(symbol, execution_price, -position_size)
                trade_value = -position_size * execution_price

            trades[symbol] = trade_value

        # Execute trades
        for symbol, trade_value in trades.items():
            price = prices[symbol]
            execution_price = self._apply_slippage(price)

            if trade_value > 0:  # Buy
                # Apply transaction cost
                trade_value *= (1 - self.transaction_cost)

                # Calculate position size
                position_size = trade_value / execution_price

                # Update positions and cash
                self.positions[symbol] += position_size
                self.cash -= trade_value

                # Record entry price (weighted average)
                if self.positions[symbol] > 0:
                    self.entry_prices[symbol] = (self.entry_prices[symbol] * (self.positions[symbol] - position_size) + execution_price * position_size) / self.positions[symbol]

                # Reset days held if new position
                if self.positions[symbol] - position_size <= 0:
                    self.days_held[symbol] = 0

            elif trade_value < 0:  # Sell
                # Calculate position size
                position_size = -trade_value / execution_price

                # Limit to available position
                position_size = min(position_size, self.positions[symbol])

                # Recalculate trade value
                trade_value = -position_size * execution_price

                # Apply transaction cost
                trade_value *= (1 - self.transaction_cost)

                # Update positions and cash
                self.positions[symbol] -= position_size
                self.cash += trade_value

                # Reset entry price if position closed
                if self.positions[symbol] <= 0:
                    self.entry_prices[symbol] = 0
                    self.days_held[symbol] = 0

            else:  # Hold
                # Increment days held
                self.days_held[symbol] += 1

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = True
        truncated = False
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                terminated = False
                break

        # Calculate portfolio value after action
        portfolio_value_after = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)

        # Calculate reward (change in portfolio value)
        reward = (portfolio_value_after - portfolio_value_before) * self.reward_scaling

        # Get next observation
        obs = self._get_obs()

        # Calculate sector exposure
        sector_exposure = self._get_sector_exposure(prices)

        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "symbol_actions": target_allocation,
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "reward": reward,
            "drawdown_pct": self.current_drawdown_pct,
            "sector_exposure": sector_exposure
        })

        # Create info dictionary
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "step": self.current_step,
            "drawdown_pct": self.current_drawdown_pct,
            "max_portfolio_value": self.max_portfolio_value,
            "sector_exposure": sector_exposure
        }

        return obs, reward, terminated, truncated, info
