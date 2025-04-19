"""
Main backtesting module.

This module provides the main backtesting functionality, tying together
data sources, strategies, and portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.backtest.portfolio import Portfolio, PortfolioManager
from src.backtest.strategy import Strategy
from src.backtest.data_source import DataSource
from src.backtest.transaction_costs import create_transaction_cost_model
from src.backtest.portfolio_metrics import calculate_portfolio_metrics, calculate_rolling_metrics
from src.backtest.visualization import (
    plot_equity_curve, plot_drawdowns, plot_returns_distribution,
    plot_monthly_returns_heatmap, plot_trade_analysis, plot_performance_metrics,
    plot_rolling_metrics
)

# Configure logging
logger = logging.getLogger(__name__)


class Backtest:
    """Main class for running backtests."""

    def __init__(
        self,
        name: str = "backtest",
        data_source: Optional[DataSource] = None,
        strategy: Optional[Strategy] = None,
        initial_cash: float = 100000.0,
        transaction_cost_model: str = "realistic",
        benchmark_symbol: Optional[str] = None,
        output_dir: str = "results"
    ):
        """
        Initialize the backtest.

        Args:
            name: Backtest name
            data_source: Data source
            strategy: Strategy
            initial_cash: Initial cash balance
            transaction_cost_model: Transaction cost model type
            benchmark_symbol: Symbol to use as benchmark
            output_dir: Directory for output files
        """
        self.name = name
        self.data_source = data_source
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.transaction_cost_model = create_transaction_cost_model(transaction_cost_model)
        self.benchmark_symbol = benchmark_symbol
        self.output_dir = output_dir

        # Create portfolio
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            transaction_cost_model=self.transaction_cost_model,
            benchmark_symbol=benchmark_symbol
        )

        # Create portfolio manager
        self.portfolio_manager = PortfolioManager()
        self.portfolio_manager.add_portfolio(name, initial_cash, self.transaction_cost_model, benchmark_symbol)

        # Results
        self.results = None

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the backtest.

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest {self.name} from {start_date} to {end_date}")

        if self.data_source is None:
            logger.error("No data source specified")
            return {}

        if self.strategy is None:
            logger.error("No strategy specified")
            return {}

        # Get data
        data = self.data_source.get_data(start_date, end_date, **kwargs)

        if data.empty:
            logger.error("No data available for backtesting")
            return {}

        # Generate signals
        signals = self.strategy.generate_signals(data, **kwargs)

        if signals.empty:
            logger.error("No signals generated")
            return {}

        # Execute signals
        for timestamp, row in signals.iterrows():
            # Get current prices
            prices = {col.replace('_price', ''): row[col] for col in row.index if col.endswith('_price')}

            # Execute signals
            self.strategy.execute_signals(self.portfolio, row, prices, timestamp, **kwargs)

            # Update portfolio
            self.portfolio.update_prices(prices, timestamp)

        # Calculate metrics
        metrics = self.portfolio.calculate_metrics()

        # Calculate rolling metrics
        rolling_metrics = self.portfolio.calculate_rolling_metrics()

        # Prepare results
        self.results = {
            'portfolio': self.portfolio,
            'metrics': metrics,
            'rolling_metrics': rolling_metrics,
            'history': self.portfolio.get_history_df(),
            'trades': self.portfolio.get_trade_history_df(),
            'positions': self.portfolio.get_positions_df(),
            'benchmark': self.portfolio.get_benchmark_history_df() if self.benchmark_symbol else None
        }

        logger.info(f"Backtest {self.name} completed")

        return self.results

    def generate_plots(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate plots for the backtest results.

        Args:
            save_dir: Directory to save plots (None to not save)

        Returns:
            Dictionary of plot names and figures
        """
        if self.results is None:
            logger.error("No backtest results available")
            return {}

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Get data
        history_df = self.results['history']
        trades_df = self.results['trades']
        benchmark_df = self.results['benchmark']
        metrics = self.results['metrics']
        rolling_metrics = self.results['rolling_metrics']

        # Prepare data for plotting
        if 'timestamp' not in history_df.columns and history_df.index.name == 'timestamp':
            history_df = history_df.reset_index()

        # Generate plots
        plots = {}

        # 1. Equity curve
        plots['equity_curve'] = plot_equity_curve(
            history_df,
            benchmark_df,
            title=f"{self.name} - Equity Curve",
            save_path=os.path.join(save_dir, "equity_curve.png") if save_dir else None
        )

        # 2. Drawdowns
        plots['drawdowns'] = plot_drawdowns(
            history_df,
            title=f"{self.name} - Drawdowns",
            save_path=os.path.join(save_dir, "drawdowns.png") if save_dir else None
        )

        # 3. Returns distribution
        plots['returns_distribution'] = plot_returns_distribution(
            history_df,
            benchmark_df,
            title=f"{self.name} - Returns Distribution",
            save_path=os.path.join(save_dir, "returns_distribution.png") if save_dir else None
        )

        # 4. Monthly returns heatmap
        plots['monthly_returns'] = plot_monthly_returns_heatmap(
            history_df,
            title=f"{self.name} - Monthly Returns",
            save_path=os.path.join(save_dir, "monthly_returns.png") if save_dir else None
        )

        # 5. Trade analysis
        if not trades_df.empty:
            plots['trade_analysis'] = plot_trade_analysis(
                trades_df,
                title=f"{self.name} - Trade Analysis",
                save_path=os.path.join(save_dir, "trade_analysis.png") if save_dir else None
            )

        # 6. Performance metrics
        plots['performance_metrics'] = plot_performance_metrics(
            metrics,
            None,  # No benchmark metrics for now
            title=f"{self.name} - Performance Metrics",
            save_path=os.path.join(save_dir, "performance_metrics.png") if save_dir else None
        )

        # 7. Rolling metrics
        if not rolling_metrics.empty:
            plots['rolling_metrics'] = plot_rolling_metrics(
                rolling_metrics,
                title=f"{self.name} - Rolling Metrics",
                save_path=os.path.join(save_dir, "rolling_metrics.png") if save_dir else None
            )

        return plots

    def generate_report(self, output_file: Optional[str] = None, include_plots: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive backtest report.

        Args:
            output_file: File to save the report (None to not save)
            include_plots: Whether to include plots in the report

        Returns:
            Dictionary with report data
        """
        if self.results is None:
            logger.error("No backtest results available")
            return {}

        # Create report directory if needed
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Get data
        history_df = self.results['history']
        trades_df = self.results['trades']
        positions_df = self.results['positions']
        metrics = self.results['metrics']

        # Prepare report
        report = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'initial_cash': self.initial_cash,
                'transaction_cost_model': self.transaction_cost_model.name if self.transaction_cost_model else 'none',
                'benchmark_symbol': self.benchmark_symbol,
                'strategy': self.strategy.name if self.strategy else 'none'
            },
            'metrics': metrics,
            'summary': {
                'initial_value': self.initial_cash,
                'final_value': history_df['total_value'].iloc[-1] if not history_df.empty else 0,
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': len(trades_df)
            }
        }

        # Add plots if requested
        if include_plots:
            # Generate plots in memory
            plots_dir = os.path.join(self.output_dir, 'plots', self.name)
            plots = self.generate_plots(plots_dir)

            # Add plot paths to report
            report['plots'] = {
                name: os.path.join('plots', self.name, f"{name}.png")
                for name in plots
            }

        # Save report if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")

        return report

    def compare(self, other_backtest: 'Backtest') -> Dict[str, Any]:
        """
        Compare this backtest with another backtest.

        Args:
            other_backtest: Another Backtest instance

        Returns:
            Dictionary with comparison results
        """
        if self.results is None or other_backtest.results is None:
            logger.error("Both backtests must have results")
            return {}

        # Get metrics
        metrics1 = self.results['metrics']
        metrics2 = other_backtest.results['metrics']

        # Select key metrics for comparison
        key_metrics = [
            'total_return',
            'annualized_return',
            'annualized_volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'calmar_ratio',
            'win_rate',
            'profit_factor'
        ]

        # Prepare comparison
        comparison = {
            'backtest1': self.name,
            'backtest2': other_backtest.name,
            'metrics': {}
        }

        # Compare metrics
        for metric in key_metrics:
            value1 = metrics1.get(metric, 0)
            value2 = metrics2.get(metric, 0)
            diff = value1 - value2
            pct_diff = diff / abs(value2) if value2 != 0 else float('inf')

            comparison['metrics'][metric] = {
                'backtest1': value1,
                'backtest2': value2,
                'difference': diff,
                'pct_difference': pct_diff
            }

        # Compare returns
        history1 = self.results['history']
        history2 = other_backtest.results['history']

        if not history1.empty and not history2.empty:
            # Calculate correlation of returns
            returns1 = history1['total_value'].pct_change().dropna()
            returns2 = history2['total_value'].pct_change().dropna()

            # Align returns
            if isinstance(returns1.index, pd.DatetimeIndex) and isinstance(returns2.index, pd.DatetimeIndex):
                returns1, returns2 = returns1.align(returns2, join='inner')

                if len(returns1) > 0:
                    correlation = returns1.corr(returns2)
                    comparison['returns_correlation'] = correlation

        return comparison


class BacktestRunner:
    """Class for running multiple backtests and comparing results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the backtest runner.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.backtests = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def add_backtest(self, backtest: Backtest) -> None:
        """
        Add a backtest.

        Args:
            backtest: Backtest instance
        """
        self.backtests[backtest.name] = backtest

    def run_all(self, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run all backtests.

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments

        Returns:
            Dictionary of backtest names and results
        """
        results = {}

        for name, backtest in self.backtests.items():
            logger.info(f"Running backtest {name}")
            results[name] = backtest.run(start_date, end_date, **kwargs)

        return results

    def generate_comparison_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comparison report for all backtests.

        Args:
            output_file: File to save the report (None to not save)

        Returns:
            Dictionary with comparison report data
        """
        if not self.backtests:
            logger.error("No backtests available")
            return {}

        # Create report directory if needed
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Prepare report
        report = {
            'timestamp': datetime.now().isoformat(),
            'backtests': list(self.backtests.keys()),
            'metrics': {},
            'comparisons': {}
        }

        # Get metrics for each backtest
        for name, backtest in self.backtests.items():
            if backtest.results is not None:
                report['metrics'][name] = backtest.results['metrics']

        # Compare backtests
        backtest_names = list(self.backtests.keys())
        for i in range(len(backtest_names)):
            for j in range(i + 1, len(backtest_names)):
                name1 = backtest_names[i]
                name2 = backtest_names[j]

                backtest1 = self.backtests[name1]
                backtest2 = self.backtests[name2]

                comparison = backtest1.compare(backtest2)
                report['comparisons'][f"{name1}_vs_{name2}"] = comparison

        # Save report if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Comparison report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving comparison report: {e}")

        return report
