"""
Example script demonstrating the backtesting framework.

This script shows how to use the backtesting framework to run a simple
moving average crossover strategy on historical stock data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.backtest.backtest import Backtest, BacktestRunner
from src.backtest.strategy import MovingAverageCrossoverStrategy, MeanReversionStrategy, MultiAssetMomentumStrategy
from src.backtest.data_source import YahooFinanceDataSource, CSVDataSource
from src.backtest.transaction_costs import create_transaction_cost_model

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_backtest():
    """Run a single backtest with a moving average crossover strategy."""
    # Define parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    initial_cash = 100000.0
    
    # Create data source
    data_source = YahooFinanceDataSource(symbols=symbols, cache_dir='data/cache')
    
    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        symbols=symbols,
        short_window=50,
        long_window=200,
        position_size=0.1
    )
    
    # Create transaction cost model
    transaction_cost_model = create_transaction_cost_model(
        model_type='realistic',
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.001,    # 0.1%
        fixed_commission=1.0    # $1 per trade
    )
    
    # Create backtest
    backtest = Backtest(
        name='ma_crossover_backtest',
        data_source=data_source,
        strategy=strategy,
        initial_cash=initial_cash,
        transaction_cost_model='realistic',
        benchmark_symbol='SPY',
        output_dir='results'
    )
    
    # Run backtest
    results = backtest.run(start_date, end_date)
    
    # Generate plots
    plots = backtest.generate_plots(save_dir='results/plots/ma_crossover_backtest')
    
    # Generate report
    report = backtest.generate_report(output_file='results/reports/ma_crossover_backtest.json')
    
    # Print summary
    print("\nBacktest Summary:")
    print("=================")
    print(f"Initial Cash: ${initial_cash:.2f}")
    print(f"Final Value: ${report['summary']['final_value']:.2f}")
    print(f"Total Return: {report['summary']['total_return']:.2%}")
    print(f"Annualized Return: {report['summary']['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['summary']['max_drawdown']:.2%}")
    print(f"Win Rate: {report['summary']['win_rate']:.2%}")
    print(f"Profit Factor: {report['summary']['profit_factor']:.2f}")
    print(f"Total Trades: {report['summary']['total_trades']}")
    
    return backtest


def run_multiple_backtests():
    """Run multiple backtests with different strategies and compare results."""
    # Define parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    initial_cash = 100000.0
    
    # Create data source
    data_source = YahooFinanceDataSource(symbols=symbols, cache_dir='data/cache')
    
    # Create backtest runner
    runner = BacktestRunner(output_dir='results')
    
    # Create and add backtests
    
    # 1. Moving Average Crossover
    ma_strategy = MovingAverageCrossoverStrategy(
        symbols=symbols,
        short_window=50,
        long_window=200,
        position_size=0.1
    )
    
    ma_backtest = Backtest(
        name='ma_crossover',
        data_source=data_source,
        strategy=ma_strategy,
        initial_cash=initial_cash,
        transaction_cost_model='realistic',
        benchmark_symbol='SPY',
        output_dir='results'
    )
    
    runner.add_backtest(ma_backtest)
    
    # 2. Mean Reversion
    mr_strategy = MeanReversionStrategy(
        symbols=symbols,
        window=20,
        num_std=2.0,
        position_size=0.1
    )
    
    mr_backtest = Backtest(
        name='mean_reversion',
        data_source=data_source,
        strategy=mr_strategy,
        initial_cash=initial_cash,
        transaction_cost_model='realistic',
        benchmark_symbol='SPY',
        output_dir='results'
    )
    
    runner.add_backtest(mr_backtest)
    
    # 3. Multi-Asset Momentum
    mom_strategy = MultiAssetMomentumStrategy(
        symbols=symbols,
        lookback_period=90,
        top_n=3,
        rebalance_frequency=30,
        position_size=0.3
    )
    
    mom_backtest = Backtest(
        name='momentum',
        data_source=data_source,
        strategy=mom_strategy,
        initial_cash=initial_cash,
        transaction_cost_model='realistic',
        benchmark_symbol='SPY',
        output_dir='results'
    )
    
    runner.add_backtest(mom_backtest)
    
    # Run all backtests
    results = runner.run_all(start_date, end_date)
    
    # Generate comparison report
    comparison_report = runner.generate_comparison_report(output_file='results/reports/strategy_comparison.json')
    
    # Print summary
    print("\nStrategy Comparison:")
    print("====================")
    
    for name, metrics in comparison_report['metrics'].items():
        print(f"\n{name}:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    return runner


def main():
    """Main function."""
    # Create output directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    
    # Run single backtest
    print("\nRunning single backtest...")
    backtest = run_single_backtest()
    
    # Run multiple backtests
    print("\nRunning multiple backtests...")
    runner = run_multiple_backtests()
    
    print("\nBacktesting completed. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
