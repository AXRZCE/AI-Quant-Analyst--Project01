"""
Strategy classes for backtesting.

This module provides base classes and implementations for trading strategies
to be used with the backtesting framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.backtest.portfolio import Portfolio

# Configure logging
logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "base_strategy"):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals from data.
        
        Args:
            data: DataFrame with market data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    @abstractmethod
    def execute_signals(self, portfolio: Portfolio, signals: pd.DataFrame, prices: Dict[str, float], timestamp: datetime, **kwargs) -> None:
        """
        Execute trading signals.
        
        Args:
            portfolio: Portfolio to execute signals on
            signals: DataFrame with trading signals
            prices: Dictionary of current prices
            timestamp: Current timestamp
            **kwargs: Additional arguments
        """
        pass
    
    def run(self, portfolio: Portfolio, data_source: Any, start_date: datetime, end_date: datetime, **kwargs) -> None:
        """
        Run the strategy on historical data.
        
        Args:
            portfolio: Portfolio to run the strategy on
            data_source: Data source for historical data
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
        """
        logger.info(f"Running strategy {self.name} from {start_date} to {end_date}")
        
        # Get historical data
        data = data_source.get_data(start_date, end_date, **kwargs)
        
        if data.empty:
            logger.warning("No data available for backtesting")
            return
        
        # Generate signals
        signals = self.generate_signals(data, **kwargs)
        
        # Execute signals
        for timestamp, row in signals.iterrows():
            # Get current prices
            prices = {col.replace('_price', ''): row[col] for col in row.index if col.endswith('_price')}
            
            # Execute signals
            self.execute_signals(portfolio, row, prices, timestamp, **kwargs)
            
            # Update portfolio
            portfolio.update_prices(prices, timestamp)
        
        logger.info(f"Strategy {self.name} completed")


class MovingAverageCrossoverStrategy(Strategy):
    """Moving average crossover strategy."""
    
    def __init__(
        self,
        name: str = "ma_crossover",
        symbols: List[str] = None,
        short_window: int = 50,
        long_window: int = 200,
        position_size: float = 0.1
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            short_window: Short moving average window
            long_window: Long moving average window
            position_size: Position size as a fraction of portfolio value
        """
        super().__init__(name)
        self.symbols = symbols or []
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: DataFrame with market data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trading signals
        """
        signals = data.copy()
        
        # Calculate moving averages for each symbol
        for symbol in self.symbols:
            price_col = f"{symbol}_price"
            
            if price_col not in signals.columns:
                logger.warning(f"Price column not found for symbol: {symbol}")
                continue
            
            # Calculate moving averages
            signals[f"{symbol}_ma_short"] = signals[price_col].rolling(window=self.short_window).mean()
            signals[f"{symbol}_ma_long"] = signals[price_col].rolling(window=self.long_window).mean()
            
            # Generate signals
            signals[f"{symbol}_signal"] = 0
            signals.loc[signals[f"{symbol}_ma_short"] > signals[f"{symbol}_ma_long"], f"{symbol}_signal"] = 1
            signals.loc[signals[f"{symbol}_ma_short"] < signals[f"{symbol}_ma_long"], f"{symbol}_signal"] = -1
            
            # Generate position changes
            signals[f"{symbol}_position_change"] = signals[f"{symbol}_signal"].diff()
        
        # Drop rows with NaN values
        signals = signals.dropna()
        
        return signals
    
    def execute_signals(self, portfolio: Portfolio, signals: pd.DataFrame, prices: Dict[str, float], timestamp: datetime, **kwargs) -> None:
        """
        Execute trading signals.
        
        Args:
            portfolio: Portfolio to execute signals on
            signals: Series with trading signals for the current timestamp
            prices: Dictionary of current prices
            timestamp: Current timestamp
            **kwargs: Additional arguments
        """
        for symbol in self.symbols:
            signal_col = f"{symbol}_signal"
            position_change_col = f"{symbol}_position_change"
            
            if signal_col not in signals or position_change_col not in signals:
                continue
            
            signal = signals[signal_col]
            position_change = signals[position_change_col]
            
            if position_change == 0:
                # No change in position
                continue
            
            if symbol not in prices:
                logger.warning(f"Price not available for symbol: {symbol}")
                continue
            
            price = prices[symbol]
            
            # Calculate position size
            portfolio_value = portfolio.total_value()
            position_value = portfolio_value * self.position_size
            quantity = position_value / price
            
            # Get current position
            position = portfolio.get_position(symbol)
            
            if position_change > 0:
                # Buy signal
                if position.quantity <= 0:
                    # Buy only if we don't already have a long position
                    portfolio.buy(symbol, quantity, price, timestamp)
            elif position_change < 0:
                # Sell signal
                if position.quantity > 0:
                    # Sell only if we have a long position
                    portfolio.sell(symbol, position.quantity, price, timestamp)


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy based on Bollinger Bands."""
    
    def __init__(
        self,
        name: str = "mean_reversion",
        symbols: List[str] = None,
        window: int = 20,
        num_std: float = 2.0,
        position_size: float = 0.1
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            window: Lookback window for calculating mean and standard deviation
            num_std: Number of standard deviations for Bollinger Bands
            position_size: Position size as a fraction of portfolio value
        """
        super().__init__(name)
        self.symbols = symbols or []
        self.window = window
        self.num_std = num_std
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: DataFrame with market data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trading signals
        """
        signals = data.copy()
        
        # Calculate Bollinger Bands for each symbol
        for symbol in self.symbols:
            price_col = f"{symbol}_price"
            
            if price_col not in signals.columns:
                logger.warning(f"Price column not found for symbol: {symbol}")
                continue
            
            # Calculate moving average and standard deviation
            signals[f"{symbol}_ma"] = signals[price_col].rolling(window=self.window).mean()
            signals[f"{symbol}_std"] = signals[price_col].rolling(window=self.window).std()
            
            # Calculate Bollinger Bands
            signals[f"{symbol}_upper_band"] = signals[f"{symbol}_ma"] + self.num_std * signals[f"{symbol}_std"]
            signals[f"{symbol}_lower_band"] = signals[f"{symbol}_ma"] - self.num_std * signals[f"{symbol}_std"]
            
            # Calculate z-score
            signals[f"{symbol}_z_score"] = (signals[price_col] - signals[f"{symbol}_ma"]) / signals[f"{symbol}_std"]
            
            # Generate signals
            signals[f"{symbol}_signal"] = 0
            signals.loc[signals[f"{symbol}_z_score"] < -self.num_std, f"{symbol}_signal"] = 1  # Buy when price is below lower band
            signals.loc[signals[f"{symbol}_z_score"] > self.num_std, f"{symbol}_signal"] = -1  # Sell when price is above upper band
            
            # Generate position changes
            signals[f"{symbol}_position_change"] = signals[f"{symbol}_signal"].diff()
        
        # Drop rows with NaN values
        signals = signals.dropna()
        
        return signals
    
    def execute_signals(self, portfolio: Portfolio, signals: pd.DataFrame, prices: Dict[str, float], timestamp: datetime, **kwargs) -> None:
        """
        Execute trading signals.
        
        Args:
            portfolio: Portfolio to execute signals on
            signals: Series with trading signals for the current timestamp
            prices: Dictionary of current prices
            timestamp: Current timestamp
            **kwargs: Additional arguments
        """
        for symbol in self.symbols:
            signal_col = f"{symbol}_signal"
            position_change_col = f"{symbol}_position_change"
            
            if signal_col not in signals or position_change_col not in signals:
                continue
            
            signal = signals[signal_col]
            position_change = signals[position_change_col]
            
            if position_change == 0:
                # No change in position
                continue
            
            if symbol not in prices:
                logger.warning(f"Price not available for symbol: {symbol}")
                continue
            
            price = prices[symbol]
            
            # Calculate position size
            portfolio_value = portfolio.total_value()
            position_value = portfolio_value * self.position_size
            quantity = position_value / price
            
            # Get current position
            position = portfolio.get_position(symbol)
            
            if position_change > 0:
                # Buy signal
                if position.quantity <= 0:
                    # Buy only if we don't already have a long position
                    portfolio.buy(symbol, quantity, price, timestamp)
            elif position_change < 0:
                # Sell signal
                if position.quantity > 0:
                    # Sell only if we have a long position
                    portfolio.sell(symbol, position.quantity, price, timestamp)


class MultiAssetMomentumStrategy(Strategy):
    """Multi-asset momentum strategy."""
    
    def __init__(
        self,
        name: str = "multi_asset_momentum",
        symbols: List[str] = None,
        lookback_period: int = 90,
        top_n: int = 3,
        rebalance_frequency: int = 30,
        position_size: float = 0.3
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            lookback_period: Lookback period for calculating momentum
            top_n: Number of top momentum assets to hold
            rebalance_frequency: Frequency of rebalancing in days
            position_size: Position size per asset as a fraction of portfolio value
        """
        super().__init__(name)
        self.symbols = symbols or []
        self.lookback_period = lookback_period
        self.top_n = min(top_n, len(self.symbols)) if self.symbols else top_n
        self.rebalance_frequency = rebalance_frequency
        self.position_size = position_size
        self.last_rebalance_date = None
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on momentum.
        
        Args:
            data: DataFrame with market data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trading signals
        """
        signals = data.copy()
        
        # Calculate momentum for each symbol
        for symbol in self.symbols:
            price_col = f"{symbol}_price"
            
            if price_col not in signals.columns:
                logger.warning(f"Price column not found for symbol: {symbol}")
                continue
            
            # Calculate momentum (return over lookback period)
            signals[f"{symbol}_momentum"] = signals[price_col].pct_change(self.lookback_period)
        
        # Calculate rebalance days
        signals['rebalance_day'] = False
        
        # Set first day as rebalance day
        if not signals.empty:
            signals.loc[signals.index[0], 'rebalance_day'] = True
        
        # Set rebalance days based on frequency
        prev_date = None
        for date in signals.index:
            if prev_date is None:
                prev_date = date
                continue
            
            days_diff = (date - prev_date).days
            if days_diff >= self.rebalance_frequency:
                signals.loc[date, 'rebalance_day'] = True
                prev_date = date
        
        # Generate signals on rebalance days
        rebalance_days = signals[signals['rebalance_day']].index
        
        for date in rebalance_days:
            # Get momentum values for all symbols
            momentum_cols = [f"{symbol}_momentum" for symbol in self.symbols if f"{symbol}_momentum" in signals.columns]
            
            if not momentum_cols:
                continue
            
            # Get momentum values for the current date
            momentum_values = signals.loc[date, momentum_cols]
            
            # Sort symbols by momentum
            sorted_symbols = momentum_values.sort_values(ascending=False)
            
            # Select top N symbols
            top_symbols = sorted_symbols.index[:self.top_n]
            top_symbols = [col.replace('_momentum', '') for col in top_symbols]
            
            # Generate signals
            for symbol in self.symbols:
                signals.loc[date, f"{symbol}_signal"] = 1 if symbol in top_symbols else -1
        
        # Forward fill signals
        signal_cols = [f"{symbol}_signal" for symbol in self.symbols]
        signals[signal_cols] = signals[signal_cols].ffill()
        
        # Generate position changes
        for symbol in self.symbols:
            signal_col = f"{symbol}_signal"
            
            if signal_col in signals.columns:
                signals[f"{symbol}_position_change"] = signals[signal_col].diff()
        
        # Drop rows with NaN values
        signals = signals.dropna()
        
        return signals
    
    def execute_signals(self, portfolio: Portfolio, signals: pd.DataFrame, prices: Dict[str, float], timestamp: datetime, **kwargs) -> None:
        """
        Execute trading signals.
        
        Args:
            portfolio: Portfolio to execute signals on
            signals: Series with trading signals for the current timestamp
            prices: Dictionary of current prices
            timestamp: Current timestamp
            **kwargs: Additional arguments
        """
        # Check if it's a rebalance day
        if not signals.get('rebalance_day', False):
            return
        
        # Update last rebalance date
        self.last_rebalance_date = timestamp
        
        # Get current portfolio value
        portfolio_value = portfolio.total_value()
        
        # Calculate target position for each symbol
        target_positions = {}
        
        for symbol in self.symbols:
            signal_col = f"{symbol}_signal"
            
            if signal_col not in signals:
                continue
            
            signal = signals[signal_col]
            
            if signal > 0:
                # Long position
                if symbol not in prices:
                    logger.warning(f"Price not available for symbol: {symbol}")
                    continue
                
                price = prices[symbol]
                target_value = portfolio_value * self.position_size
                target_quantity = target_value / price
                target_positions[symbol] = target_quantity
            else:
                # No position
                target_positions[symbol] = 0
        
        # Execute trades to reach target positions
        for symbol, target_quantity in target_positions.items():
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            position = portfolio.get_position(symbol)
            current_quantity = position.quantity
            
            if target_quantity > current_quantity:
                # Buy
                quantity_to_buy = target_quantity - current_quantity
                portfolio.buy(symbol, quantity_to_buy, price, timestamp)
            elif target_quantity < current_quantity:
                # Sell
                quantity_to_sell = current_quantity - target_quantity
                portfolio.sell(symbol, quantity_to_sell, price, timestamp)
