"""
Visualization tools for backtesting results.

This module provides functions to create various visualizations for analyzing
backtesting results, including equity curves, drawdowns, and trade analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import calendar

# Import portfolio metrics
from src.backtest.portfolio_metrics import calculate_monthly_returns

# Configure logging
logger = logging.getLogger(__name__)


def plot_equity_curve(
    portfolio_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    title: str = "Portfolio Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = False,
    show_drawdowns: bool = True,
    drawdown_threshold: float = -0.1,  # 10% drawdown
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio equity curve with optional benchmark comparison.

    Args:
        portfolio_df: DataFrame with portfolio history (must have 'timestamp' and 'portfolio_value' columns)
        benchmark_df: Optional DataFrame with benchmark data (must have 'timestamp' and 'close' columns)
        title: Plot title
        figsize: Figure size
        log_scale: Whether to use logarithmic scale for y-axis
        show_drawdowns: Whether to highlight drawdown periods
        drawdown_threshold: Threshold for highlighting drawdowns
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if 'timestamp' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
        logger.error("Portfolio DataFrame must have 'timestamp' and 'portfolio_value' columns")
        return plt.figure()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot portfolio equity curve
    ax.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio', linewidth=2)

    # Plot benchmark if provided
    if benchmark_df is not None and 'timestamp' in benchmark_df.columns and 'close' in benchmark_df.columns:
        # Align benchmark with portfolio dates
        benchmark_aligned = benchmark_df.set_index('timestamp').reindex(portfolio_df['timestamp'])

        # Normalize benchmark to same starting value
        benchmark_normalized = benchmark_aligned['close'] / benchmark_aligned['close'].iloc[0] * portfolio_df['portfolio_value'].iloc[0]

        ax.plot(portfolio_df['timestamp'], benchmark_normalized, label='Benchmark', alpha=0.7, linewidth=1.5)

    # Highlight drawdown periods if requested
    if show_drawdowns and len(portfolio_df) > 1:
        # Calculate drawdowns
        portfolio_df = portfolio_df.copy()
        portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cum_return'] = (1 + portfolio_df['return']).cumprod()
        portfolio_df['cum_max'] = portfolio_df['cum_return'].cummax()
        portfolio_df['drawdown'] = portfolio_df['cum_return'] / portfolio_df['cum_max'] - 1

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None

        for i, (idx, row) in enumerate(portfolio_df.iterrows()):
            if row['drawdown'] <= drawdown_threshold and not in_drawdown:
                # Start of significant drawdown
                in_drawdown = True
                start_idx = i
            elif row['drawdown'] > drawdown_threshold and in_drawdown:
                # End of significant drawdown
                in_drawdown = False
                drawdown_periods.append((start_idx, i))

        # If still in drawdown at the end
        if in_drawdown:
            drawdown_periods.append((start_idx, len(portfolio_df) - 1))

        # Highlight drawdown periods
        for start, end in drawdown_periods:
            ax.axvspan(
                portfolio_df['timestamp'].iloc[start],
                portfolio_df['timestamp'].iloc[end],
                alpha=0.2,
                color='red',
                label='_nolegend_'
            )

    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_drawdowns(
    portfolio_df: pd.DataFrame,
    title: str = "Portfolio Drawdowns",
    figsize: Tuple[int, int] = (12, 6),
    top_n: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio drawdowns.

    Args:
        portfolio_df: DataFrame with portfolio history (must have 'timestamp' and 'portfolio_value' columns)
        title: Plot title
        figsize: Figure size
        top_n: Number of top drawdowns to annotate
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if 'timestamp' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
        logger.error("Portfolio DataFrame must have 'timestamp' and 'portfolio_value' columns")
        return plt.figure()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdowns
    portfolio_df = portfolio_df.copy()
    portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
    portfolio_df['cum_return'] = (1 + portfolio_df['return']).cumprod()
    portfolio_df['cum_max'] = portfolio_df['cum_return'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['cum_return'] / portfolio_df['cum_max'] - 1) * 100  # Convert to percentage

    # Plot drawdowns
    ax.fill_between(
        portfolio_df['timestamp'],
        portfolio_df['drawdown'],
        0,
        where=(portfolio_df['drawdown'] < 0),
        color='red',
        alpha=0.3,
        label='Drawdown'
    )

    # Find top drawdowns
    drawdown_periods = []
    in_drawdown = False
    start_idx = None
    max_drawdown = 0

    for i, (idx, row) in enumerate(portfolio_df.iterrows()):
        if row['drawdown'] < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = i
            max_drawdown = 0

        if in_drawdown:
            # Update maximum drawdown
            max_drawdown = min(max_drawdown, row['drawdown'])

        if row['drawdown'] >= 0 and in_drawdown:
            # End of drawdown
            in_drawdown = False
            drawdown_periods.append((start_idx, i - 1, max_drawdown))

    # If still in drawdown at the end
    if in_drawdown:
        drawdown_periods.append((start_idx, len(portfolio_df) - 1, max_drawdown))

    # Sort drawdowns by magnitude
    drawdown_periods.sort(key=lambda x: x[2])

    # Annotate top drawdowns
    for i, (start, end, max_dd) in enumerate(drawdown_periods[:top_n]):
        # Find index of maximum drawdown
        max_dd_idx = portfolio_df['drawdown'].iloc[start:end+1].idxmin()
        max_dd_date = portfolio_df.loc[max_dd_idx, 'timestamp']
        max_dd_value = portfolio_df.loc[max_dd_idx, 'drawdown']

        # Annotate
        ax.annotate(
            f"{max_dd_value:.1f}%",
            xy=(max_dd_date, max_dd_value),
            xytext=(max_dd_date, max_dd_value - 5),
            arrowprops=dict(arrowstyle="->", color='black'),
            ha='center',
            fontsize=9
        )

    # Set y-axis limits
    ax.set_ylim(min(portfolio_df['drawdown'].min() * 1.1, -5), 5)

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Add labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_returns_distribution(
    portfolio_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    period: str = 'D',  # 'D' for daily, 'W' for weekly, 'M' for monthly
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of returns.

    Args:
        portfolio_df: DataFrame with portfolio history (must have 'timestamp' and 'portfolio_value' columns)
        benchmark_df: Optional DataFrame with benchmark data (must have 'timestamp' and 'close' columns)
        title: Plot title
        figsize: Figure size
        bins: Number of bins for histogram
        period: Return period ('D' for daily, 'W' for weekly, 'M' for monthly)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if 'timestamp' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
        logger.error("Portfolio DataFrame must have 'timestamp' and 'portfolio_value' columns")
        return plt.figure()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate returns
    portfolio_df = portfolio_df.copy()
    portfolio_df = portfolio_df.set_index('timestamp')

    # Resample based on period
    if period == 'W':
        portfolio_value_resampled = portfolio_df['portfolio_value'].resample('W').last()
    elif period == 'M':
        portfolio_value_resampled = portfolio_df['portfolio_value'].resample('M').last()
    else:  # Default to daily
        portfolio_value_resampled = portfolio_df['portfolio_value']

    portfolio_returns = portfolio_value_resampled.pct_change().dropna()

    # Calculate benchmark returns if provided
    if benchmark_df is not None and 'timestamp' in benchmark_df.columns and 'close' in benchmark_df.columns:
        benchmark_df = benchmark_df.copy()
        benchmark_df = benchmark_df.set_index('timestamp')

        # Resample based on period
        if period == 'W':
            benchmark_value_resampled = benchmark_df['close'].resample('W').last()
        elif period == 'M':
            benchmark_value_resampled = benchmark_df['close'].resample('M').last()
        else:  # Default to daily
            benchmark_value_resampled = benchmark_df['close']

        benchmark_returns = benchmark_value_resampled.pct_change().dropna()

        # Align returns
        portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    # Plot portfolio returns distribution
    sns.histplot(portfolio_returns, bins=bins, kde=True, alpha=0.6, label='Portfolio', ax=ax)

    # Plot benchmark returns distribution if provided
    if benchmark_df is not None:
        sns.histplot(benchmark_returns, bins=bins, kde=True, alpha=0.4, label='Benchmark', ax=ax)

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)

    # Add mean and std dev lines for portfolio
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()

    ax.axvline(x=mean_return, color='blue', linestyle='-', alpha=0.7, label=f'Mean: {mean_return:.2%}')
    ax.axvline(x=mean_return + std_return, color='blue', linestyle=':', alpha=0.7, label=f'+1 Std Dev: {mean_return + std_return:.2%}')
    ax.axvline(x=mean_return - std_return, color='blue', linestyle=':', alpha=0.7, label=f'-1 Std Dev: {mean_return - std_return:.2%}')

    # Add labels
    period_label = 'Daily' if period == 'D' else 'Weekly' if period == 'W' else 'Monthly'
    ax.set_title(f"{title} ({period_label})")
    ax.set_xlabel(f'{period_label} Returns')
    ax.set_ylabel('Frequency')

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_monthly_returns_heatmap(
    portfolio_df: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn",  # Red for negative, green for positive
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot monthly returns as a heatmap.

    Args:
        portfolio_df: DataFrame with portfolio history (must have 'timestamp' and 'portfolio_value' columns)
        title: Plot title
        figsize: Figure size
        cmap: Colormap for heatmap
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if 'timestamp' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
        logger.error("Portfolio DataFrame must have 'timestamp' and 'portfolio_value' columns")
        return plt.figure()

    # Calculate monthly returns
    portfolio_df = portfolio_df.copy()
    portfolio_df = portfolio_df.set_index('timestamp')
    monthly_returns = calculate_monthly_returns(portfolio_df['portfolio_value'].pct_change().dropna())

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        monthly_returns.iloc[:, :-1],  # Exclude 'Annual' column
        annot=True,
        fmt='.1%',
        cmap=cmap,
        center=0,
        linewidths=1,
        cbar=True,
        ax=ax
    )

    # Add annual returns
    annual_returns = monthly_returns['Annual'].dropna()
    for i, (year, annual_return) in enumerate(annual_returns.items()):
        ax.text(
            monthly_returns.shape[1] - 0.5,  # Position after last month
            i + 0.5,  # Center in row
            f"{annual_return:.1%}",
            ha='center',
            va='center',
            fontweight='bold'
        )

    # Set month names as column labels
    ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)] + ['Annual'])

    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel('Year')

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_trade_analysis(
    trades_df: pd.DataFrame,
    title: str = "Trade Analysis",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive trade analysis.

    Args:
        trades_df: DataFrame with trade history
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if len(trades_df) == 0:
        logger.error("Trades DataFrame is empty")
        return plt.figure()

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Trade PnL
    ax1 = fig.add_subplot(gs[0, 0])
    trades_with_pnl = trades_df[trades_df['pnl'] != 0].copy()

    if len(trades_with_pnl) > 0:
        colors = ["green" if pnl > 0 else "red" for pnl in trades_with_pnl['pnl']]
        ax1.bar(range(len(trades_with_pnl)), trades_with_pnl['pnl'], color=colors)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title("Trade PnL")
        ax1.set_xlabel("Trade #")
        ax1.set_ylabel("PnL ($)")
        ax1.grid(True, alpha=0.3)

        # Add cumulative PnL line
        ax1_twin = ax1.twinx()
        cumulative_pnl = trades_with_pnl['pnl'].cumsum()
        ax1_twin.plot(range(len(trades_with_pnl)), cumulative_pnl, color='blue', linestyle='-', alpha=0.7)
        ax1_twin.set_ylabel("Cumulative PnL ($)", color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
    else:
        ax1.text(0.5, 0.5, "No trades with PnL", ha='center', va='center')
        ax1.set_title("Trade PnL")
        ax1.axis("off")

    # 2. Trade Returns
    ax2 = fig.add_subplot(gs[0, 1])
    if 'return' in trades_with_pnl.columns and len(trades_with_pnl) > 0:
        colors = ["green" if ret > 0 else "red" for ret in trades_with_pnl['return']]
        ax2.bar(range(len(trades_with_pnl)), trades_with_pnl['return'], color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title("Trade Returns")
        ax2.set_xlabel("Trade #")
        ax2.set_ylabel("Return (%)")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No trade returns data", ha='center', va='center')
        ax2.set_title("Trade Returns")
        ax2.axis("off")

    # 3. Trade Holding Periods
    ax3 = fig.add_subplot(gs[1, 0])
    if 'holding_period' in trades_with_pnl.columns and len(trades_with_pnl) > 0:
        sns.histplot(trades_with_pnl['holding_period'], bins=20, kde=True, ax=ax3)
        ax3.set_title("Holding Periods Distribution")
        ax3.set_xlabel("Holding Period (days)")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No holding period data", ha='center', va='center')
        ax3.set_title("Holding Periods Distribution")
        ax3.axis("off")

    # 4. PnL by Symbol
    ax4 = fig.add_subplot(gs[1, 1])
    if 'symbol' in trades_with_pnl.columns and len(trades_with_pnl) > 0:
        symbol_pnl = trades_with_pnl.groupby('symbol')['pnl'].sum().sort_values()
        colors = ["green" if pnl > 0 else "red" for pnl in symbol_pnl]
        symbol_pnl.plot(kind='barh', color=colors, ax=ax4)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title("PnL by Symbol")
        ax4.set_xlabel("PnL ($)")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No symbol data", ha='center', va='center')
        ax4.set_title("PnL by Symbol")
        ax4.axis("off")

    # 5. Win/Loss Statistics
    ax5 = fig.add_subplot(gs[2, 0])
    if len(trades_with_pnl) > 0:
        winning_trades = trades_with_pnl[trades_with_pnl['pnl'] > 0]
        losing_trades = trades_with_pnl[trades_with_pnl['pnl'] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_count = len(trades_with_pnl)

        win_rate = win_count / total_count if total_count > 0 else 0

        # Create pie chart
        ax5.pie(
            [win_count, loss_count],
            labels=['Wins', 'Losses'],
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90
        )
        ax5.set_title(f"Win/Loss Ratio: {win_count}/{loss_count} ({win_rate:.1%})")
    else:
        ax5.text(0.5, 0.5, "No trade data", ha='center', va='center')
        ax5.set_title("Win/Loss Statistics")
        ax5.axis("off")

    # 6. Average PnL Statistics
    ax6 = fig.add_subplot(gs[2, 1])
    if len(trades_with_pnl) > 0:
        winning_trades = trades_with_pnl[trades_with_pnl['pnl'] > 0]
        losing_trades = trades_with_pnl[trades_with_pnl['pnl'] < 0]

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        # Create bar chart
        ax6.bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss], color=['green', 'red'])
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Calculate profit factor
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1e-9
        profit_factor = total_profit / total_loss

        ax6.set_title(f"Avg PnL - Profit Factor: {profit_factor:.2f}")
        ax6.set_ylabel("PnL ($)")
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "No trade data", ha='center', va='center')
        ax6.set_title("Average PnL Statistics")
        ax6.axis("off")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_performance_metrics(
    metrics: Dict[str, float],
    benchmark_metrics: Optional[Dict[str, float]] = None,
    title: str = "Performance Metrics",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance metrics comparison.

    Args:
        metrics: Dictionary of portfolio metrics
        benchmark_metrics: Optional dictionary of benchmark metrics
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Select key metrics to display
    key_metrics = [
        'annualized_return',
        'annualized_volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'win_rate',
        'profit_factor'
    ]

    # Filter metrics
    metrics_to_plot = {k: v for k, v in metrics.items() if k in key_metrics}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    # Plot portfolio metrics
    rects1 = ax.bar(x - width/2 if benchmark_metrics else x, list(metrics_to_plot.values()), width, label='Portfolio')

    # Plot benchmark metrics if provided
    if benchmark_metrics:
        benchmark_to_plot = {k: benchmark_metrics.get(k, 0) for k in key_metrics}
        rects2 = ax.bar(x + width/2, list(benchmark_to_plot.values()), width, label='Benchmark')

    # Add labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('_', ' ').title() for k in metrics_to_plot.keys()], rotation=45)
    ax.legend()

    # Format y-axis for percentage metrics
    percentage_metrics = ['annualized_return', 'annualized_volatility', 'max_drawdown', 'win_rate']
    for i, metric in enumerate(metrics_to_plot.keys()):
        if metric in percentage_metrics:
            # Add percentage sign to the value
            ax.text(i - width/2 if benchmark_metrics else i, metrics_to_plot[metric] * 1.05,
                   f"{metrics_to_plot[metric]:.1%}", ha='center', va='bottom')

            if benchmark_metrics and metric in benchmark_to_plot:
                ax.text(i + width/2, benchmark_to_plot[metric] * 1.05,
                       f"{benchmark_to_plot[metric]:.1%}", ha='center', va='bottom')
        else:
            # Add regular value
            ax.text(i - width/2 if benchmark_metrics else i, metrics_to_plot[metric] * 1.05,
                   f"{metrics_to_plot[metric]:.2f}", ha='center', va='bottom')

            if benchmark_metrics and metric in benchmark_to_plot:
                ax.text(i + width/2, benchmark_to_plot[metric] * 1.05,
                       f"{benchmark_to_plot[metric]:.2f}", ha='center', va='bottom')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_rolling_metrics(
    rolling_metrics: pd.DataFrame,
    title: str = "Rolling Performance Metrics",
    figsize: Tuple[int, int] = (12, 10),
    window: int = 63,  # ~3 months of trading days
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics.

    Args:
        rolling_metrics: DataFrame with rolling metrics
        title: Plot title
        figsize: Figure size
        window: Rolling window size in days
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if len(rolling_metrics) == 0:
        logger.error("Rolling metrics DataFrame is empty")
        return plt.figure()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 1. Rolling Returns
    if 'rolling_return' in rolling_metrics.columns:
        axes[0].plot(rolling_metrics.index, rolling_metrics['rolling_return'], color='blue')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_title(f"{window}-day Rolling Return")
        axes[0].set_ylabel("Return")
        axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[0].grid(True, alpha=0.3)

    # 2. Rolling Sharpe Ratio
    if 'rolling_sharpe' in rolling_metrics.columns:
        axes[1].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], color='green')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title(f"{window}-day Rolling Sharpe Ratio")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].grid(True, alpha=0.3)

    # 3. Rolling Maximum Drawdown
    if 'rolling_max_drawdown' in rolling_metrics.columns:
        axes[2].plot(rolling_metrics.index, rolling_metrics['rolling_max_drawdown'], color='red')
        axes[2].set_title(f"{window}-day Rolling Maximum Drawdown")
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Max Drawdown")
        axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[2].grid(True, alpha=0.3)

    # Format x-axis as dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    plt.xticks(rotation=45)

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig