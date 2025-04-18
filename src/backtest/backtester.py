"""
Backtesting framework for evaluating trading strategies.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for evaluating trading strategies.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model: Optional[Any] = None,
        capital: float = 100_000,
        commission: float = 0.001,  # 0.1% commission per trade
        slippage: float = 0.001,    # 0.1% slippage per trade
        prediction_column: str = "pred"
    ):
        """
        Initialize the backtester.

        Args:
            df: DataFrame with price data and predictions
            model: Optional model for making predictions
            capital: Initial capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            prediction_column: Name of the prediction column
        """
        self.df = df.copy()
        self.model = model
        self.capital = capital
        self.commission = commission
        self.slippage = slippage
        self.prediction_column = prediction_column

        # Ensure DataFrame is sorted by timestamp
        if "timestamp" in self.df.columns:
            self.df = self.df.sort_values("timestamp")

        # Generate predictions if model is provided and predictions don't exist
        if model is not None and prediction_column not in self.df.columns:
            self._generate_predictions()

    def _generate_predictions(self) -> None:
        """
        Generate predictions using the provided model.
        """
        logger.info("Generating predictions using the model")

        # Prepare features
        feature_cols = [col for col in self.df.columns if col not in ["timestamp", "symbol", "date", "label"]]
        X = self.df[feature_cols]

        # Generate predictions
        self.df[self.prediction_column] = self.model.predict(X)

        logger.info(f"Generated {len(self.df)} predictions")

    def run(
        self,
        strategy: Optional[Callable[[pd.Series], int]] = None,
        threshold: float = 0.0,
        max_position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        Run the backtest.

        Args:
            strategy: Custom strategy function that takes a row and returns a position size (-1 to 1)
            threshold: Prediction threshold for entering positions
            max_position_size: Maximum position size as a fraction of capital
            stop_loss: Optional stop loss as a fraction of entry price
            take_profit: Optional take profit as a fraction of entry price

        Returns:
            Final capital, trade history, and performance metrics
        """
        logger.info(f"Running backtest with initial capital ${self.capital:.2f}")

        # Initialize variables
        cash = self.capital
        positions = {}  # symbol -> (quantity, entry_price, entry_time)
        trade_history = []
        portfolio_history = []

        # Default strategy: if pred > threshold, go long; if pred < -threshold, go short
        if strategy is None:
            def default_strategy(row):
                if row[self.prediction_column] > threshold:
                    return max_position_size  # Long
                elif row[self.prediction_column] < -threshold:
                    return -max_position_size  # Short
                else:
                    return 0  # No position

            strategy = default_strategy

        # Group by timestamp to process all symbols at each timestamp
        for timestamp, group in self.df.groupby("timestamp"):
            # Calculate portfolio value at this timestamp
            portfolio_value = cash
            for symbol, (qty, entry_price, _) in positions.items():
                # Find current price for this symbol
                symbol_data = group[group["symbol"] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]["close"]
                    portfolio_value += qty * current_price

            # Process each symbol
            for _, row in group.iterrows():
                symbol = row["symbol"]
                current_price = row["close"]

                # Check if we have a position in this symbol
                if symbol in positions:
                    qty, entry_price, entry_time = positions[symbol]

                    # Check stop loss
                    if stop_loss is not None:
                        if qty > 0 and current_price <= entry_price * (1 - stop_loss):  # Long position stop loss
                            # Close position
                            exit_price = current_price * (1 - self.slippage)  # Account for slippage
                            cash += qty * exit_price * (1 - self.commission)  # Account for commission

                            # Record trade
                            trade_history.append({
                                "timestamp": timestamp,
                                "symbol": symbol,
                                "action": "STOP_LOSS",
                                "qty": -qty,
                                "price": exit_price,
                                "value": qty * exit_price,
                                "commission": qty * exit_price * self.commission,
                                "pnl": qty * (exit_price - entry_price),
                                "return": (exit_price / entry_price - 1) * 100,
                                "holding_period": (timestamp - entry_time).total_seconds() / 86400  # in days
                            })

                            # Remove position
                            del positions[symbol]
                            continue

                        elif qty < 0 and current_price >= entry_price * (1 + stop_loss):  # Short position stop loss
                            # Close position
                            exit_price = current_price * (1 + self.slippage)  # Account for slippage
                            cash += -qty * exit_price * (1 - self.commission)  # Account for commission

                            # Record trade
                            trade_history.append({
                                "timestamp": timestamp,
                                "symbol": symbol,
                                "action": "STOP_LOSS",
                                "qty": -qty,
                                "price": exit_price,
                                "value": -qty * exit_price,
                                "commission": -qty * exit_price * self.commission,
                                "pnl": -qty * (entry_price - exit_price),
                                "return": (entry_price / exit_price - 1) * 100,
                                "holding_period": (timestamp - entry_time).total_seconds() / 86400  # in days
                            })

                            # Remove position
                            del positions[symbol]
                            continue

                    # Check take profit
                    if take_profit is not None:
                        if qty > 0 and current_price >= entry_price * (1 + take_profit):  # Long position take profit
                            # Close position
                            exit_price = current_price * (1 - self.slippage)  # Account for slippage
                            cash += qty * exit_price * (1 - self.commission)  # Account for commission

                            # Record trade
                            trade_history.append({
                                "timestamp": timestamp,
                                "symbol": symbol,
                                "action": "TAKE_PROFIT",
                                "qty": -qty,
                                "price": exit_price,
                                "value": qty * exit_price,
                                "commission": qty * exit_price * self.commission,
                                "pnl": qty * (exit_price - entry_price),
                                "return": (exit_price / entry_price - 1) * 100,
                                "holding_period": (timestamp - entry_time).total_seconds() / 86400  # in days
                            })

                            # Remove position
                            del positions[symbol]
                            continue

                        elif qty < 0 and current_price <= entry_price * (1 - take_profit):  # Short position take profit
                            # Close position
                            exit_price = current_price * (1 + self.slippage)  # Account for slippage
                            cash += -qty * exit_price * (1 - self.commission)  # Account for commission

                            # Record trade
                            trade_history.append({
                                "timestamp": timestamp,
                                "symbol": symbol,
                                "action": "TAKE_PROFIT",
                                "qty": -qty,
                                "price": exit_price,
                                "value": -qty * exit_price,
                                "commission": -qty * exit_price * self.commission,
                                "pnl": -qty * (entry_price - exit_price),
                                "return": (entry_price / exit_price - 1) * 100,
                                "holding_period": (timestamp - entry_time).total_seconds() / 86400  # in days
                            })

                            # Remove position
                            del positions[symbol]
                            continue

                # Get position size from strategy (-1 to 1)
                position_size = strategy(row)

                # Skip if no position change
                if symbol in positions and abs(position_size) < 1e-6:
                    continue

                # Close existing position if direction changes or position size is zero
                if symbol in positions:
                    qty, entry_price, entry_time = positions[symbol]

                    # Check if we need to close the position
                    if (qty > 0 and position_size <= 0) or (qty < 0 and position_size >= 0) or position_size == 0:
                        # Close position
                        exit_price = current_price
                        if qty > 0:
                            exit_price *= (1 - self.slippage)  # Account for slippage when selling
                        else:
                            exit_price *= (1 + self.slippage)  # Account for slippage when buying to cover

                        cash += abs(qty) * exit_price * (1 - self.commission)  # Account for commission

                        # Record trade
                        trade_history.append({
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "action": "CLOSE",
                            "qty": -qty,
                            "price": exit_price,
                            "value": abs(qty) * exit_price,
                            "commission": abs(qty) * exit_price * self.commission,
                            "pnl": qty * (exit_price - entry_price) if qty > 0 else -qty * (entry_price - exit_price),
                            "return": (exit_price / entry_price - 1) * 100 if qty > 0 else (entry_price / exit_price - 1) * 100,
                            "holding_period": (timestamp - entry_time).total_seconds() / 86400  # in days
                        })

                        # Remove position
                        del positions[symbol]

                # Open new position if position size is non-zero
                if position_size != 0 and symbol not in positions:
                    # Calculate position value
                    position_value = portfolio_value * abs(position_size)

                    # Calculate quantity
                    entry_price = current_price
                    if position_size > 0:
                        entry_price *= (1 + self.slippage)  # Account for slippage when buying
                    else:
                        entry_price *= (1 - self.slippage)  # Account for slippage when shorting

                    qty = position_value / entry_price
                    if position_size < 0:
                        qty = -qty  # Negative quantity for short positions

                    # Check if we have enough cash
                    if position_size > 0 and cash < position_value * (1 + self.commission):
                        # Not enough cash for long position
                        continue

                    # Update cash
                    if position_size > 0:
                        cash -= position_value * (1 + self.commission)  # Account for commission

                    # Record position
                    positions[symbol] = (qty, entry_price, timestamp)

                    # Record trade
                    trade_history.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "action": "BUY" if position_size > 0 else "SHORT",
                        "qty": qty,
                        "price": entry_price,
                        "value": position_value,
                        "commission": position_value * self.commission,
                        "pnl": 0,
                        "return": 0,
                        "holding_period": 0
                    })

            # Record portfolio value
            portfolio_history.append({
                "timestamp": timestamp,
                "cash": cash,
                "positions_value": portfolio_value - cash,
                "portfolio_value": portfolio_value
            })

        # Close any remaining positions at the end
        for symbol, (qty, entry_price, entry_time) in list(positions.items()):
            # Find last price for this symbol
            symbol_data = self.df[self.df["symbol"] == symbol].iloc[-1]
            exit_price = symbol_data["close"]

            if qty > 0:
                exit_price *= (1 - self.slippage)  # Account for slippage when selling
            else:
                exit_price *= (1 + self.slippage)  # Account for slippage when buying to cover

            cash += abs(qty) * exit_price * (1 - self.commission)  # Account for commission

            # Record trade
            trade_history.append({
                "timestamp": symbol_data["timestamp"],
                "symbol": symbol,
                "action": "FINAL_CLOSE",
                "qty": -qty,
                "price": exit_price,
                "value": abs(qty) * exit_price,
                "commission": abs(qty) * exit_price * self.commission,
                "pnl": qty * (exit_price - entry_price) if qty > 0 else -qty * (entry_price - exit_price),
                "return": (exit_price / entry_price - 1) * 100 if qty > 0 else (entry_price / exit_price - 1) * 100,
                "holding_period": (symbol_data["timestamp"] - entry_time).total_seconds() / 86400  # in days
            })

        # Convert to DataFrames
        trades_df = pd.DataFrame(trade_history)
        portfolio_df = pd.DataFrame(portfolio_history)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_df, trades_df)

        logger.info(f"Backtest completed with final capital ${cash:.2f}")
        logger.info(f"Total trades: {len(trades_df)}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Annualized return: {metrics['annualized_return']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")

        return cash, trades_df, portfolio_df

    def _calculate_performance_metrics(
        self,
        portfolio_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            portfolio_df: DataFrame with portfolio history
            trades_df: DataFrame with trade history

        Returns:
            Dictionary of performance metrics
        """
        # Calculate daily returns
        portfolio_df["daily_return"] = portfolio_df["portfolio_value"].pct_change()

        # Calculate total return
        initial_value = portfolio_df["portfolio_value"].iloc[0]
        final_value = portfolio_df["portfolio_value"].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        # Calculate annualized return
        days = (portfolio_df["timestamp"].iloc[-1] - portfolio_df["timestamp"].iloc[0]).total_seconds() / 86400
        try:
            annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
            # Cap at a reasonable value to avoid overflow
            annualized_return = min(annualized_return, 1000000)
        except (OverflowError, RuntimeWarning):
            annualized_return = float('inf') if final_value > initial_value else float('-inf')

        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1 / 365) - 1
        excess_returns = portfolio_df["daily_return"] - daily_risk_free
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized

        # Calculate max drawdown
        portfolio_df["cumulative_return"] = (1 + portfolio_df["daily_return"]).cumprod()
        portfolio_df["cumulative_max"] = portfolio_df["cumulative_return"].cummax()
        portfolio_df["drawdown"] = (portfolio_df["cumulative_return"] / portfolio_df["cumulative_max"] - 1) * 100
        max_drawdown = portfolio_df["drawdown"].min()

        # Calculate win rate
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df["pnl"] > 0]
            win_rate = len(winning_trades) / len(trades_df) * 100

            # Calculate average profit/loss
            avg_profit = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades_df[trades_df["pnl"] < 0]
            avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0

            # Calculate profit factor
            total_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1e-9
            profit_factor = total_profit / total_loss
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": len(trades_df),
            "trading_days": days
        }

    def plot_results(
        self,
        portfolio_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Plot backtest results.

        Args:
            portfolio_df: DataFrame with portfolio history
            trades_df: DataFrame with trade history
            benchmark_df: Optional DataFrame with benchmark data
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        # Plot portfolio value
        plt.subplot(3, 1, 1)
        plt.plot(portfolio_df["timestamp"], portfolio_df["portfolio_value"], label="Portfolio Value")

        # Plot benchmark if provided
        if benchmark_df is not None:
            # Normalize benchmark to same starting value
            benchmark_df["normalized_value"] = benchmark_df["close"] / benchmark_df["close"].iloc[0] * portfolio_df["portfolio_value"].iloc[0]
            plt.plot(benchmark_df["timestamp"], benchmark_df["normalized_value"], label="Benchmark", alpha=0.7)

        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)

        # Plot drawdown
        plt.subplot(3, 1, 2)
        plt.fill_between(portfolio_df["timestamp"], portfolio_df["drawdown"], 0, color="red", alpha=0.3)
        plt.plot(portfolio_df["timestamp"], portfolio_df["drawdown"], color="red", label="Drawdown")
        plt.title("Portfolio Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)

        # Plot trade PnL
        plt.subplot(3, 1, 3)

        # Filter out trades with zero PnL
        trades_with_pnl = trades_df[trades_df["pnl"] != 0]

        if len(trades_with_pnl) > 0:
            colors = ["green" if pnl > 0 else "red" for pnl in trades_with_pnl["pnl"]]
            plt.bar(range(len(trades_with_pnl)), trades_with_pnl["pnl"], color=colors)
            plt.title("Trade PnL")
            plt.xlabel("Trade #")
            plt.ylabel("PnL ($)")
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No trades with PnL", horizontalalignment="center", verticalalignment="center")
            plt.title("Trade PnL")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

def load_model(model_path: str) -> Any:
    """
    Load a trained model from a file.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")

    # Load model package
    model_package = joblib.load(model_path)

    # Extract model
    if isinstance(model_package, dict) and "model" in model_package:
        model = model_package["model"]

        # Log metadata if available
        if "metadata" in model_package:
            logger.info(f"Model metadata: {model_package['metadata']}")

        return model
    else:
        # If not a package, assume it's the model directly
        return model_package

def prepare_backtest_data(
    df: pd.DataFrame,
    model: Any,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare data for backtesting by generating predictions.

    Args:
        df: DataFrame with price data
        model: Model for generating predictions
        feature_cols: List of feature columns to use

    Returns:
        DataFrame with predictions
    """
    logger.info("Preparing data for backtesting")

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # If feature columns not provided, use all columns except timestamp, symbol, date, and label
    if feature_cols is None:
        feature_cols = [col for col in df_copy.columns if col not in ["timestamp", "symbol", "date", "label", "pred"]]

    # Generate predictions
    X = df_copy[feature_cols]
    df_copy["pred"] = model.predict(X)

    logger.info(f"Generated {len(df_copy)} predictions")

    return df_copy

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save results")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--threshold", type=float, default=0.0, help="Prediction threshold")

    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data_path)

    # Load model
    model = load_model(args.model_path)

    # Prepare data
    df = prepare_backtest_data(df, model)

    # Run backtest
    backtester = Backtester(df, model=None, capital=args.capital, commission=args.commission)
    final_capital, trades_df, portfolio_df = backtester.run(threshold=args.threshold)

    # Plot results
    backtester.plot_results(portfolio_df, trades_df)

    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        # Save trades and portfolio history
        trades_df.to_csv(f"{args.output_path}_trades.csv", index=False)
        portfolio_df.to_csv(f"{args.output_path}_portfolio.csv", index=False)

        logger.info(f"Results saved to {args.output_path}_trades.csv and {args.output_path}_portfolio.csv")
