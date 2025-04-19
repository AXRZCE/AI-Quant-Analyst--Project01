"""
Portfolio class for multi-asset backtesting.

This module provides a Portfolio class to manage positions across multiple assets,
calculate portfolio value, and track performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from src.backtest.transaction_costs import TransactionCostModel, create_transaction_cost_model
from src.backtest.portfolio_metrics import calculate_portfolio_metrics, calculate_rolling_metrics

# Configure logging
logger = logging.getLogger(__name__)


class Position:
    """Class representing a position in a single asset."""
    
    def __init__(self, symbol: str, quantity: float = 0, avg_price: float = 0.0):
        """
        Initialize a position.
        
        Args:
            symbol: Asset symbol
            quantity: Quantity held
            avg_price: Average entry price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.cost_basis = quantity * avg_price if quantity != 0 else 0.0
        self.realized_pnl = 0.0
        self.trades = []
        self.last_price = avg_price
    
    def update_price(self, price: float) -> None:
        """
        Update the position with a new price.
        
        Args:
            price: New price
        """
        self.last_price = price
    
    def market_value(self) -> float:
        """
        Calculate the market value of the position.
        
        Returns:
            Market value
        """
        return self.quantity * self.last_price
    
    def unrealized_pnl(self) -> float:
        """
        Calculate the unrealized PnL of the position.
        
        Returns:
            Unrealized PnL
        """
        return self.market_value() - self.cost_basis
    
    def total_pnl(self) -> float:
        """
        Calculate the total PnL of the position.
        
        Returns:
            Total PnL (realized + unrealized)
        """
        return self.realized_pnl + self.unrealized_pnl()
    
    def add(self, quantity: float, price: float) -> float:
        """
        Add to the position.
        
        Args:
            quantity: Quantity to add
            price: Price of the new shares
            
        Returns:
            Cost of the transaction
        """
        if quantity <= 0:
            logger.warning(f"Cannot add non-positive quantity: {quantity}")
            return 0.0
        
        # Calculate transaction cost
        transaction_cost = quantity * price
        
        # Update position
        new_total_quantity = self.quantity + quantity
        new_cost_basis = self.cost_basis + transaction_cost
        
        # Update average price
        if new_total_quantity > 0:
            self.avg_price = new_cost_basis / new_total_quantity
        
        # Update quantity and cost basis
        self.quantity = new_total_quantity
        self.cost_basis = new_cost_basis
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.now(),
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'cost': transaction_cost
        })
        
        return transaction_cost
    
    def remove(self, quantity: float, price: float) -> Tuple[float, float]:
        """
        Remove from the position.
        
        Args:
            quantity: Quantity to remove
            price: Price of the shares
            
        Returns:
            Tuple of (proceeds from the transaction, realized PnL)
        """
        if quantity <= 0:
            logger.warning(f"Cannot remove non-positive quantity: {quantity}")
            return 0.0, 0.0
        
        if quantity > self.quantity:
            logger.warning(f"Cannot remove more than held: {quantity} > {self.quantity}")
            quantity = self.quantity
        
        # Calculate transaction proceeds
        transaction_proceeds = quantity * price
        
        # Calculate realized PnL
        realized_pnl = (price - self.avg_price) * quantity
        self.realized_pnl += realized_pnl
        
        # Update position
        self.cost_basis -= quantity * self.avg_price
        self.quantity -= quantity
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.now(),
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'proceeds': transaction_proceeds,
            'realized_pnl': realized_pnl
        })
        
        return transaction_proceeds, realized_pnl
    
    def __str__(self) -> str:
        """String representation of the position."""
        return (f"Position({self.symbol}, quantity={self.quantity}, "
                f"avg_price={self.avg_price:.2f}, market_value={self.market_value():.2f}, "
                f"unrealized_pnl={self.unrealized_pnl():.2f}, realized_pnl={self.realized_pnl:.2f})")


class Portfolio:
    """Class representing a portfolio of positions across multiple assets."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        benchmark_symbol: Optional[str] = None
    ):
        """
        Initialize a portfolio.
        
        Args:
            initial_cash: Initial cash balance
            transaction_cost_model: Model for transaction costs
            benchmark_symbol: Symbol to use as benchmark
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.transaction_cost_model = transaction_cost_model or create_transaction_cost_model("simple")
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_initial_price = None
        self.benchmark_price = None
        
        # History
        self.history = []
        self.trade_history = []
        self.benchmark_history = []
    
    def get_position(self, symbol: str) -> Position:
        """
        Get a position by symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def market_value(self) -> float:
        """
        Calculate the total market value of the portfolio.
        
        Returns:
            Total market value
        """
        return sum(position.market_value() for position in self.positions.values())
    
    def total_value(self) -> float:
        """
        Calculate the total value of the portfolio (cash + market value).
        
        Returns:
            Total portfolio value
        """
        return self.cash + self.market_value()
    
    def unrealized_pnl(self) -> float:
        """
        Calculate the total unrealized PnL of the portfolio.
        
        Returns:
            Total unrealized PnL
        """
        return sum(position.unrealized_pnl() for position in self.positions.values())
    
    def realized_pnl(self) -> float:
        """
        Calculate the total realized PnL of the portfolio.
        
        Returns:
            Total realized PnL
        """
        return sum(position.realized_pnl for position in self.positions.values())
    
    def total_pnl(self) -> float:
        """
        Calculate the total PnL of the portfolio.
        
        Returns:
            Total PnL (realized + unrealized)
        """
        return self.realized_pnl() + self.unrealized_pnl()
    
    def update_prices(self, prices: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Update prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> price
            timestamp: Optional timestamp for the update
        """
        timestamp = timestamp or datetime.now()
        
        # Update position prices
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
            
            # Update benchmark price
            if symbol == self.benchmark_symbol:
                if self.benchmark_initial_price is None:
                    self.benchmark_initial_price = price
                self.benchmark_price = price
                
                # Record benchmark history
                self.benchmark_history.append({
                    'timestamp': timestamp,
                    'price': price,
                    'return': price / self.benchmark_initial_price - 1 if self.benchmark_initial_price else 0.0
                })
        
        # Record portfolio history
        self.history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'market_value': self.market_value(),
            'total_value': self.total_value(),
            'unrealized_pnl': self.unrealized_pnl(),
            'realized_pnl': self.realized_pnl(),
            'total_pnl': self.total_pnl(),
            'return': self.total_value() / self.initial_cash - 1
        })
    
    def buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """
        Buy an asset.
        
        Args:
            symbol: Asset symbol
            quantity: Quantity to buy
            price: Price per unit
            timestamp: Optional timestamp for the trade
            
        Returns:
            Tuple of (success, cost)
        """
        timestamp = timestamp or datetime.now()
        
        if quantity <= 0:
            logger.warning(f"Cannot buy non-positive quantity: {quantity}")
            return False, 0.0
        
        # Calculate transaction cost
        base_cost = quantity * price
        transaction_cost = self.transaction_cost_model.calculate_cost(price, quantity, symbol=symbol, timestamp=timestamp)
        adjusted_price = self.transaction_cost_model.adjust_price(price, quantity, symbol=symbol, timestamp=timestamp)
        total_cost = quantity * adjusted_price + transaction_cost
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for purchase: {total_cost} > {self.cash}")
            return False, 0.0
        
        # Update position
        position = self.get_position(symbol)
        position.add(quantity, adjusted_price)
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'adjusted_price': adjusted_price,
            'base_cost': base_cost,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        return True, total_cost
    
    def sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, float, float]:
        """
        Sell an asset.
        
        Args:
            symbol: Asset symbol
            quantity: Quantity to sell
            price: Price per unit
            timestamp: Optional timestamp for the trade
            
        Returns:
            Tuple of (success, proceeds, realized_pnl)
        """
        timestamp = timestamp or datetime.now()
        
        if quantity <= 0:
            logger.warning(f"Cannot sell non-positive quantity: {quantity}")
            return False, 0.0, 0.0
        
        # Get position
        if symbol not in self.positions:
            logger.warning(f"No position for symbol: {symbol}")
            return False, 0.0, 0.0
        
        position = self.positions[symbol]
        
        if position.quantity < quantity:
            logger.warning(f"Insufficient quantity for sale: {position.quantity} < {quantity}")
            quantity = position.quantity
        
        # Calculate transaction cost
        adjusted_price = self.transaction_cost_model.adjust_price(price, -quantity, symbol=symbol, timestamp=timestamp)
        transaction_cost = self.transaction_cost_model.calculate_cost(price, -quantity, symbol=symbol, timestamp=timestamp)
        
        # Update position
        proceeds, realized_pnl = position.remove(quantity, adjusted_price)
        
        # Update cash
        net_proceeds = proceeds - transaction_cost
        self.cash += net_proceeds
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'adjusted_price': adjusted_price,
            'base_proceeds': proceeds,
            'transaction_cost': transaction_cost,
            'net_proceeds': net_proceeds,
            'realized_pnl': realized_pnl,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        return True, net_proceeds, realized_pnl
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Get portfolio history as a DataFrame.
        
        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.history)
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        return pd.DataFrame(self.trade_history)
    
    def get_benchmark_history_df(self) -> pd.DataFrame:
        """
        Get benchmark history as a DataFrame.
        
        Returns:
            DataFrame with benchmark history
        """
        return pd.DataFrame(self.benchmark_history)
    
    def get_positions_df(self) -> pd.DataFrame:
        """
        Get current positions as a DataFrame.
        
        Returns:
            DataFrame with current positions
        """
        positions_data = []
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                positions_data.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'last_price': position.last_price,
                    'market_value': position.market_value(),
                    'cost_basis': position.cost_basis,
                    'unrealized_pnl': position.unrealized_pnl(),
                    'realized_pnl': position.realized_pnl,
                    'total_pnl': position.total_pnl()
                })
        
        return pd.DataFrame(positions_data)
    
    def get_portfolio_returns(self) -> pd.Series:
        """
        Get portfolio returns as a Series.
        
        Returns:
            Series of portfolio returns
        """
        history_df = self.get_history_df()
        
        if len(history_df) < 2:
            return pd.Series()
        
        # Calculate returns
        history_df = history_df.set_index('timestamp')
        portfolio_values = history_df['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        return returns
    
    def get_benchmark_returns(self) -> pd.Series:
        """
        Get benchmark returns as a Series.
        
        Returns:
            Series of benchmark returns
        """
        benchmark_df = self.get_benchmark_history_df()
        
        if len(benchmark_df) < 2:
            return pd.Series()
        
        # Calculate returns
        benchmark_df = benchmark_df.set_index('timestamp')
        benchmark_prices = benchmark_df['price']
        returns = benchmark_prices.pct_change().dropna()
        
        return returns
    
    def calculate_metrics(self, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods in a year
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Get returns
        portfolio_returns = self.get_portfolio_returns()
        benchmark_returns = self.get_benchmark_returns()
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(
            portfolio_returns,
            benchmark_returns if not benchmark_returns.empty else None,
            risk_free_rate,
            periods_per_year
        )
        
        # Add portfolio-specific metrics
        history_df = self.get_history_df()
        if not history_df.empty:
            metrics['initial_value'] = self.initial_cash
            metrics['final_value'] = history_df['total_value'].iloc[-1]
            metrics['absolute_return'] = metrics['final_value'] / metrics['initial_value'] - 1
            metrics['cash'] = history_df['cash'].iloc[-1]
            metrics['market_value'] = history_df['market_value'].iloc[-1]
        
        # Add trade metrics
        trade_df = self.get_trade_history_df()
        if not trade_df.empty:
            metrics['total_trades'] = len(trade_df)
            metrics['buy_trades'] = len(trade_df[trade_df['action'] == 'buy'])
            metrics['sell_trades'] = len(trade_df[trade_df['action'] == 'sell'])
            metrics['total_volume'] = trade_df['quantity'].sum()
            metrics['total_commission'] = trade_df['transaction_cost'].sum()
        
        return metrics
    
    def calculate_rolling_metrics(
        self,
        window: int = 63,  # ~3 months of trading days
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling portfolio metrics.
        
        Args:
            window: Rolling window size
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods in a year
            
        Returns:
            DataFrame of rolling metrics
        """
        # Get returns
        portfolio_returns = self.get_portfolio_returns()
        benchmark_returns = self.get_benchmark_returns()
        
        # Calculate rolling metrics
        return calculate_rolling_metrics(
            portfolio_returns,
            benchmark_returns if not benchmark_returns.empty else None,
            risk_free_rate,
            window,
            periods_per_year
        )
    
    def __str__(self) -> str:
        """String representation of the portfolio."""
        return (f"Portfolio(cash={self.cash:.2f}, market_value={self.market_value():.2f}, "
                f"total_value={self.total_value():.2f}, positions={len(self.positions)})")


class PortfolioManager:
    """Class for managing multiple portfolios and strategies."""
    
    def __init__(self):
        """Initialize the portfolio manager."""
        self.portfolios = {}  # name -> Portfolio
        self.strategies = {}  # name -> Strategy
        self.data_sources = {}  # name -> DataSource
    
    def add_portfolio(
        self,
        name: str,
        initial_cash: float = 100000.0,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        benchmark_symbol: Optional[str] = None
    ) -> Portfolio:
        """
        Add a portfolio.
        
        Args:
            name: Portfolio name
            initial_cash: Initial cash balance
            transaction_cost_model: Model for transaction costs
            benchmark_symbol: Symbol to use as benchmark
            
        Returns:
            Portfolio object
        """
        if name in self.portfolios:
            logger.warning(f"Portfolio already exists: {name}")
            return self.portfolios[name]
        
        portfolio = Portfolio(initial_cash, transaction_cost_model, benchmark_symbol)
        self.portfolios[name] = portfolio
        
        return portfolio
    
    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """
        Get a portfolio by name.
        
        Args:
            name: Portfolio name
            
        Returns:
            Portfolio object or None if not found
        """
        return self.portfolios.get(name)
    
    def add_strategy(self, name: str, strategy: Any) -> None:
        """
        Add a strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy object
        """
        self.strategies[name] = strategy
    
    def get_strategy(self, name: str) -> Optional[Any]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy object or None if not found
        """
        return self.strategies.get(name)
    
    def add_data_source(self, name: str, data_source: Any) -> None:
        """
        Add a data source.
        
        Args:
            name: Data source name
            data_source: Data source object
        """
        self.data_sources[name] = data_source
    
    def get_data_source(self, name: str) -> Optional[Any]:
        """
        Get a data source by name.
        
        Args:
            name: Data source name
            
        Returns:
            Data source object or None if not found
        """
        return self.data_sources.get(name)
    
    def run_backtest(
        self,
        portfolio_name: str,
        strategy_name: str,
        data_source_name: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            portfolio_name: Portfolio name
            strategy_name: Strategy name
            data_source_name: Data source name
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for the strategy
            
        Returns:
            Dictionary with backtest results
        """
        # Get portfolio, strategy, and data source
        portfolio = self.get_portfolio(portfolio_name)
        strategy = self.get_strategy(strategy_name)
        data_source = self.get_data_source(data_source_name)
        
        if not portfolio:
            logger.error(f"Portfolio not found: {portfolio_name}")
            return {}
        
        if not strategy:
            logger.error(f"Strategy not found: {strategy_name}")
            return {}
        
        if not data_source:
            logger.error(f"Data source not found: {data_source_name}")
            return {}
        
        # Run backtest
        try:
            strategy.run(portfolio, data_source, start_date, end_date, **kwargs)
            
            # Calculate metrics
            metrics = portfolio.calculate_metrics()
            
            return {
                'portfolio': portfolio,
                'metrics': metrics,
                'history': portfolio.get_history_df(),
                'trades': portfolio.get_trade_history_df(),
                'positions': portfolio.get_positions_df()
            }
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
