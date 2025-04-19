# Backtesting Framework Documentation

This document provides an overview of the enhanced backtesting framework for the AI Quant Analyst project.

## Overview

The backtesting framework is designed to evaluate trading strategies using historical market data. It includes:

1. **Transaction Costs and Slippage Models**: Realistic simulation of trading costs
2. **Portfolio-Level Metrics**: Comprehensive performance evaluation metrics
3. **Multi-Asset Support**: Ability to backtest strategies across multiple assets
4. **Visualization Tools**: Advanced visualization for performance analysis

## Components

### Transaction Cost Models

The framework includes several transaction cost models:

- **Fixed Commission**: Fixed fee per trade
- **Percentage Commission**: Fee as a percentage of trade value
- **Tiered Commission**: Fee structure based on trade value
- **Slippage Models**: Constant, volume-based, and bid-ask spread models
- **Market Impact**: Price impact based on trade size

These models can be combined to create realistic trading simulations.

### Portfolio Management

The framework includes a robust portfolio management system:

- **Position Tracking**: Track positions across multiple assets
- **Cash Management**: Track cash balance and portfolio value
- **Trade History**: Record all trades with detailed information
- **Multi-Asset Support**: Manage positions across multiple assets

### Performance Metrics

The framework calculates a comprehensive set of performance metrics:

- **Return Metrics**: Total return, annualized return, etc.
- **Risk Metrics**: Volatility, drawdown, VaR, etc.
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, etc.
- **Trading Metrics**: Win rate, profit factor, etc.
- **Benchmark-Relative Metrics**: Alpha, beta, information ratio, etc.

### Visualization Tools

The framework includes advanced visualization tools:

- **Equity Curve**: Plot portfolio value over time
- **Drawdowns**: Visualize drawdown periods
- **Returns Distribution**: Analyze return distribution
- **Monthly Returns Heatmap**: View returns by month and year
- **Trade Analysis**: Analyze trade performance
- **Performance Metrics**: Compare key performance metrics
- **Rolling Metrics**: View metrics over rolling windows

## Usage

### Basic Usage

```python
from src.backtest.backtest import Backtest
from src.backtest.strategy import MovingAverageCrossoverStrategy
from src.backtest.data_source import YahooFinanceDataSource

# Create data source
data_source = YahooFinanceDataSource(symbols=['AAPL', 'MSFT', 'GOOGL'])

# Create strategy
strategy = MovingAverageCrossoverStrategy(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    short_window=50,
    long_window=200,
    position_size=0.1
)

# Create backtest
backtest = Backtest(
    name='ma_crossover',
    data_source=data_source,
    strategy=strategy,
    initial_cash=100000.0,
    transaction_cost_model='realistic',
    benchmark_symbol='SPY'
)

# Run backtest
results = backtest.run(start_date, end_date)

# Generate plots
plots = backtest.generate_plots()

# Generate report
report = backtest.generate_report()
```

### Running Multiple Backtests

```python
from src.backtest.backtest import BacktestRunner

# Create backtest runner
runner = BacktestRunner()

# Add backtests
runner.add_backtest(backtest1)
runner.add_backtest(backtest2)
runner.add_backtest(backtest3)

# Run all backtests
results = runner.run_all(start_date, end_date)

# Generate comparison report
comparison_report = runner.generate_comparison_report()
```

## Strategies

The framework includes several built-in strategies:

### Moving Average Crossover

A strategy that generates buy signals when a short-term moving average crosses above a long-term moving average, and sell signals when it crosses below.

```python
strategy = MovingAverageCrossoverStrategy(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    short_window=50,
    long_window=200,
    position_size=0.1
)
```

### Mean Reversion

A strategy that buys when the price is significantly below its moving average and sells when it's significantly above.

```python
strategy = MeanReversionStrategy(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    window=20,
    num_std=2.0,
    position_size=0.1
)
```

### Multi-Asset Momentum

A strategy that ranks assets by their momentum and invests in the top performers.

```python
strategy = MultiAssetMomentumStrategy(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX'],
    lookback_period=90,
    top_n=3,
    rebalance_frequency=30,
    position_size=0.3
)
```

## Data Sources

The framework supports multiple data sources:

### Yahoo Finance

```python
data_source = YahooFinanceDataSource(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    cache_dir='data/cache'
)
```

### CSV Files

```python
data_source = CSVDataSource(
    data_dir='data',
    filename_template='{symbol}.csv',
    date_column='timestamp',
    price_column='close',
    symbols=['AAPL', 'MSFT', 'GOOGL']
)
```

### SQL Database

```python
data_source = SQLDataSource(
    connection_string='sqlite:///data/market_data.db',
    table_name='market_data',
    symbol_column='symbol',
    date_column='timestamp',
    price_column='close',
    symbols=['AAPL', 'MSFT', 'GOOGL']
)
```

## Extending the Framework

### Creating Custom Strategies

To create a custom strategy, inherit from the `Strategy` base class and implement the required methods:

```python
class MyCustomStrategy(Strategy):
    def __init__(self, name="my_strategy", symbols=None, **kwargs):
        super().__init__(name)
        self.symbols = symbols or []
        # Initialize other parameters
        
    def generate_signals(self, data, **kwargs):
        # Generate trading signals
        signals = data.copy()
        # Add signal generation logic
        return signals
        
    def execute_signals(self, portfolio, signals, prices, timestamp, **kwargs):
        # Execute trading signals
        # Add execution logic
```

### Creating Custom Data Sources

To create a custom data source, inherit from the `DataSource` base class and implement the required methods:

```python
class MyCustomDataSource(DataSource):
    def __init__(self, name="my_data_source", **kwargs):
        super().__init__(name)
        # Initialize parameters
        
    def get_data(self, start_date, end_date, **kwargs):
        # Get data for the specified date range
        # Add data retrieval logic
        return data
```

## Best Practices

1. **Use Realistic Transaction Costs**: Always include realistic transaction costs to avoid overfitting.
2. **Include Slippage**: Account for slippage, especially for strategies with high turnover.
3. **Use Proper Time Series Cross-Validation**: Avoid look-ahead bias by using proper time series validation.
4. **Compare Against Benchmarks**: Always compare strategy performance against relevant benchmarks.
5. **Analyze Drawdowns**: Pay attention to drawdowns, not just returns.
6. **Consider Multiple Metrics**: Don't focus solely on returns; consider risk-adjusted metrics.
7. **Test Across Different Market Conditions**: Evaluate strategies across bull, bear, and sideways markets.
8. **Perform Sensitivity Analysis**: Test strategies with different parameter values.

## Conclusion

The enhanced backtesting framework provides a robust platform for evaluating trading strategies with realistic transaction costs, comprehensive performance metrics, multi-asset support, and advanced visualization tools. It can be easily extended with custom strategies and data sources to meet specific requirements.
