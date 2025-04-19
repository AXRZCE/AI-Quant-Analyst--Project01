"""
Reinforcement learning strategy wrapper for backtesting.

This module provides a wrapper for reinforcement learning algorithms
to be used with the backtesting framework.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import RL algorithms
from src.rl.sac import SAC

# Import PPO if available
try:
    from src.rl.ppo import PPO
except ImportError:
    # Define a placeholder PPO class for testing
    class PPO:
        def __init__(self, state_dim=None, action_dim=None, device="cpu"):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.device = device

        def select_action(self, state, evaluate=False):
            return np.zeros(self.action_dim)

        def save(self, path):
            pass

        def load(self, path):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLStrategy:
    """Reinforcement learning strategy wrapper for backtesting."""

    def __init__(
        self,
        algorithm: str = "sac",
        model_path: Optional[str] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        feature_cols: Optional[List[str]] = None,
        price_col: str = "close",
        date_col: str = "timestamp",
        symbol_col: str = "symbol",
        position_cols: Optional[List[str]] = None,
        normalize_state: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RL strategy.

        Args:
            algorithm: RL algorithm to use ("sac" or "ppo")
            model_path: Path to pre-trained model
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            feature_cols: List of feature column names
            price_col: Name of price column
            date_col: Name of date column
            symbol_col: Name of symbol column
            position_cols: List of position column names
            normalize_state: Whether to normalize state
            device: Device to use for inference
        """
        self.algorithm = algorithm.lower()
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_cols = feature_cols or []
        self.price_col = price_col
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.position_cols = position_cols or []
        self.normalize_state = normalize_state
        self.device = device

        # Initialize agent
        self.agent = None

        # Initialize state normalization parameters
        self.state_mean = None
        self.state_std = None

        # Initialize agent
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize RL agent."""
        if self.algorithm == "sac":
            if self.state_dim is None or self.action_dim is None:
                raise ValueError("state_dim and action_dim must be provided for SAC")

            self.agent = SAC(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
        elif self.algorithm == "ppo":
            if self.state_dim is None or self.action_dim is None:
                raise ValueError("state_dim and action_dim must be provided for PPO")

            self.agent = PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Load pre-trained model if provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained model.

        Args:
            model_path: Path to pre-trained model
        """
        if self.agent is None:
            raise ValueError("Agent not initialized")

        try:
            self.agent.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str) -> None:
        """
        Save model.

        Args:
            model_path: Path to save model
        """
        if self.agent is None:
            raise ValueError("Agent not initialized")

        try:
            self.agent.save(model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def fit_normalizer(self, df: pd.DataFrame) -> None:
        """
        Fit state normalizer.

        Args:
            df: DataFrame with features
        """
        if not self.normalize_state:
            return

        if not self.feature_cols:
            raise ValueError("feature_cols must be provided for normalization")

        # Extract features
        features = df[self.feature_cols].values

        # Compute mean and std
        self.state_mean = np.mean(features, axis=0)
        self.state_std = np.std(features, axis=0)
        self.state_std[self.state_std == 0] = 1.0  # Avoid division by zero

        logger.info("Fitted state normalizer")

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state.

        Args:
            state: State to normalize

        Returns:
            Normalized state
        """
        if not self.normalize_state or self.state_mean is None or self.state_std is None:
            return state

        return (state - self.state_mean) / self.state_std

    def get_state(self, df: pd.DataFrame, idx: int, symbol: str) -> np.ndarray:
        """
        Get state from DataFrame.

        Args:
            df: DataFrame with features
            idx: Current index
            symbol: Current symbol

        Returns:
            State vector
        """
        # Filter DataFrame for current symbol
        symbol_df = df[df[self.symbol_col] == symbol]

        if idx >= len(symbol_df):
            raise ValueError(f"Index {idx} out of bounds for symbol {symbol}")

        # Extract features
        if self.feature_cols:
            state = symbol_df.iloc[idx][self.feature_cols].values
        else:
            # Use all numeric columns except date and symbol
            numeric_cols = symbol_df.select_dtypes(include=[np.number]).columns
            exclude_cols = [self.date_col, self.symbol_col]
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            state = symbol_df.iloc[idx][feature_cols].values

        # Add position information if available
        if self.position_cols:
            position_values = symbol_df.iloc[idx][self.position_cols].values
            state = np.concatenate([state, position_values])

        # Normalize state
        if self.normalize_state and self.state_mean is not None and self.state_std is not None:
            state = self.normalize_state(state)

        return state.astype(np.float32)

    def predict(self, df: pd.DataFrame, idx: int, symbols: List[str]) -> Dict[str, float]:
        """
        Predict actions for current state.

        Args:
            df: DataFrame with features
            idx: Current index
            symbols: List of symbols

        Returns:
            Dictionary of symbol to action mapping
        """
        if self.agent is None:
            raise ValueError("Agent not initialized")

        actions = {}

        for symbol in symbols:
            try:
                # Get state
                state = self.get_state(df, idx, symbol)

                # Select action
                action = self.agent.select_action(state, evaluate=True)

                # Convert to scalar for single-asset case
                if isinstance(action, np.ndarray) and len(action) == 1:
                    action = float(action[0])

                actions[symbol] = action
            except Exception as e:
                logger.error(f"Error predicting action for {symbol}: {e}")
                actions[symbol] = 0.0

        return actions

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for backtesting.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with signals
        """
        if self.agent is None:
            raise ValueError("Agent not initialized")

        # Fit normalizer if needed
        if self.normalize_state and self.state_mean is None:
            self.fit_normalizer(df)

        # Get unique symbols
        symbols = df[self.symbol_col].unique().tolist()

        # Get unique dates
        dates = df[self.date_col].unique()

        # Initialize results
        results = []

        # Generate signals for each date
        for date_idx, date in enumerate(dates):
            date_df = df[df[self.date_col] == date]

            # Predict actions
            actions = self.predict(df, date_idx, symbols)

            # Convert actions to signals
            for symbol, action in actions.items():
                # Get price
                symbol_df = date_df[date_df[self.symbol_col] == symbol]
                if len(symbol_df) == 0:
                    continue

                price = symbol_df[self.price_col].iloc[0]

                # Convert action to signal
                if isinstance(action, np.ndarray):
                    # Multi-dimensional action
                    signal = float(action[0])  # Use first dimension as signal
                else:
                    # Scalar action
                    signal = float(action)

                # Clip signal to [-1, 1]
                signal = np.clip(signal, -1.0, 1.0)

                # Add to results
                results.append({
                    self.date_col: date,
                    self.symbol_col: symbol,
                    self.price_col: price,
                    "signal": signal,
                    "position": signal  # Position is directly proportional to signal
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Run backtest.

        Args:
            df: DataFrame with features
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction of trade value
            slippage: Slippage as a fraction of price

        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals_df = self.generate_signals(df)

        # Initialize portfolio
        portfolio = {
            "cash": initial_capital,
            "positions": {symbol: 0.0 for symbol in signals_df[self.symbol_col].unique()},
            "equity": [initial_capital],
            "returns": [0.0],
            "dates": [signals_df[self.date_col].iloc[0]],
            "trades": []
        }

        # Run backtest
        dates = signals_df[self.date_col].unique()

        for date_idx, date in enumerate(dates):
            date_df = signals_df[signals_df[self.date_col] == date]

            # Calculate portfolio value before trading
            portfolio_value_before = portfolio["cash"] + sum(
                portfolio["positions"][symbol] * date_df[date_df[self.symbol_col] == symbol][self.price_col].iloc[0]
                for symbol in portfolio["positions"]
                if symbol in date_df[self.symbol_col].values
            )

            # Execute trades
            for _, row in date_df.iterrows():
                symbol = row[self.symbol_col]
                price = row[self.price_col]
                signal = row["signal"]

                # Calculate target position
                target_position = signal * portfolio_value_before / price

                # Calculate trade size
                current_position = portfolio["positions"][symbol]
                trade_size = target_position - current_position

                # Skip small trades
                if abs(trade_size) < 0.01:
                    continue

                # Apply slippage
                execution_price = price * (1 + np.sign(trade_size) * slippage)

                # Calculate trade value
                trade_value = abs(trade_size * execution_price)

                # Apply transaction cost
                trade_cost = trade_value * transaction_cost

                # Execute trade
                if trade_size > 0:  # Buy
                    if portfolio["cash"] >= trade_value + trade_cost:
                        portfolio["positions"][symbol] += trade_size
                        portfolio["cash"] -= trade_value + trade_cost

                        # Record trade
                        portfolio["trades"].append({
                            "date": date,
                            "symbol": symbol,
                            "action": "buy",
                            "price": execution_price,
                            "size": trade_size,
                            "value": trade_value,
                            "cost": trade_cost
                        })
                elif trade_size < 0:  # Sell
                    portfolio["positions"][symbol] += trade_size
                    portfolio["cash"] += trade_value - trade_cost

                    # Record trade
                    portfolio["trades"].append({
                        "date": date,
                        "symbol": symbol,
                        "action": "sell",
                        "price": execution_price,
                        "size": -trade_size,
                        "value": trade_value,
                        "cost": trade_cost
                    })

            # Calculate portfolio value after trading
            portfolio_value_after = portfolio["cash"] + sum(
                portfolio["positions"][symbol] * date_df[date_df[self.symbol_col] == symbol][self.price_col].iloc[0]
                for symbol in portfolio["positions"]
                if symbol in date_df[self.symbol_col].values
            )

            # Record equity and return
            portfolio["equity"].append(portfolio_value_after)
            portfolio["returns"].append(
                (portfolio_value_after / portfolio["equity"][-2] - 1)
                if date_idx > 0 else 0.0
            )
            portfolio["dates"].append(date)

        # Calculate performance metrics
        returns = np.array(portfolio["returns"][1:])  # Skip initial return of 0

        # Calculate metrics
        total_return = portfolio["equity"][-1] / portfolio["equity"][0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Calculate drawdown
        equity = np.array(portfolio["equity"])
        max_equity = np.maximum.accumulate(equity)
        drawdown = (equity - max_equity) / max_equity
        max_drawdown = np.min(drawdown)

        # Calculate win rate
        trades = portfolio["trades"]
        if trades:
            buy_trades = [t for t in trades if t["action"] == "buy"]
            sell_trades = [t for t in trades if t["action"] == "sell"]

            # Match buy and sell trades
            matched_trades = []
            for buy in buy_trades:
                for sell in sell_trades:
                    if sell["date"] > buy["date"] and sell["symbol"] == buy["symbol"]:
                        profit = (sell["price"] - buy["price"]) / buy["price"]
                        matched_trades.append({
                            "buy_date": buy["date"],
                            "sell_date": sell["date"],
                            "symbol": buy["symbol"],
                            "profit": profit
                        })
                        sell_trades.remove(sell)
                        break

            # Calculate win rate
            if matched_trades:
                win_rate = sum(1 for t in matched_trades if t["profit"] > 0) / len(matched_trades)
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0

        # Create results dictionary
        results = {
            "initial_capital": initial_capital,
            "final_equity": portfolio["equity"][-1],
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(portfolio["trades"]),
            "equity_curve": portfolio["equity"],
            "returns": portfolio["returns"],
            "dates": portfolio["dates"],
            "trades": portfolio["trades"]
        }

        return results

    def plot_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot backtest results.

        Args:
            results: Dictionary with backtest results
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter

            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

            # Convert dates to datetime
            dates = pd.to_datetime(results["dates"])

            # Plot equity curve
            axes[0].plot(dates, results["equity_curve"])
            axes[0].set_title("Equity Curve")
            axes[0].set_ylabel("Equity ($)")
            axes[0].grid(True)

            # Format y-axis as currency
            axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

            # Plot returns
            axes[1].plot(dates, [0] + results["returns"][1:])
            axes[1].set_title("Daily Returns")
            axes[1].set_ylabel("Return (%)")
            axes[1].grid(True)

            # Format y-axis as percentage
            axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.2f}%"))

            # Plot drawdown
            equity = np.array(results["equity_curve"])
            max_equity = np.maximum.accumulate(equity)
            drawdown = (equity - max_equity) / max_equity

            axes[2].fill_between(dates, 0, drawdown, color="red", alpha=0.3)
            axes[2].set_title("Drawdown")
            axes[2].set_ylabel("Drawdown (%)")
            axes[2].set_xlabel("Date")
            axes[2].grid(True)

            # Format y-axis as percentage
            axes[2].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.2f}%"))

            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Add performance metrics as text
            metrics_text = (
                f"Total Return: {results['total_return']*100:.2f}%\n"
                f"Annual Return: {results['annual_return']*100:.2f}%\n"
                f"Annual Volatility: {results['annual_volatility']*100:.2f}%\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {results['max_drawdown']*100:.2f}%\n"
                f"Win Rate: {results['win_rate']*100:.2f}%\n"
                f"Number of Trades: {results['num_trades']}"
            )

            plt.figtext(0.01, 0.01, metrics_text, fontsize=10, va="bottom")

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # Show plot
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not available. Install with 'pip install matplotlib'")
        except Exception as e:
            logger.error(f"Error plotting results: {e}")


class RLStrategyFactory:
    """Factory for creating RL strategies."""

    @staticmethod
    def create_strategy(
        algorithm: str = "sac",
        model_path: Optional[str] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        feature_cols: Optional[List[str]] = None,
        price_col: str = "close",
        date_col: str = "timestamp",
        symbol_col: str = "symbol",
        position_cols: Optional[List[str]] = None,
        normalize_state: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> RLStrategy:
        """
        Create RL strategy.

        Args:
            algorithm: RL algorithm to use ("sac" or "ppo")
            model_path: Path to pre-trained model
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            feature_cols: List of feature column names
            price_col: Name of price column
            date_col: Name of date column
            symbol_col: Name of symbol column
            position_cols: List of position column names
            normalize_state: Whether to normalize state
            device: Device to use for inference

        Returns:
            RL strategy
        """
        return RLStrategy(
            algorithm=algorithm,
            model_path=model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            feature_cols=feature_cols,
            price_col=price_col,
            date_col=date_col,
            symbol_col=symbol_col,
            position_cols=position_cols,
            normalize_state=normalize_state,
            device=device
        )
