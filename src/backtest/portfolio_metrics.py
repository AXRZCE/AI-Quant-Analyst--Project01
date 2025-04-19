"""
Portfolio performance metrics for backtesting.

This module provides functions to calculate various portfolio performance metrics
including risk-adjusted returns, drawdowns, and other financial metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series, method: str = "arithmetic") -> pd.Series:
    """
    Calculate returns from a price series.
    
    Args:
        prices: Series of prices
        method: Method to calculate returns ('arithmetic' or 'log')
        
    Returns:
        Series of returns
    """
    if method == "arithmetic":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        logger.warning(f"Unknown return method: {method}, using 'arithmetic' instead")
        return prices.pct_change()


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize a series of returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized return
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate compound return
    compound_return = (1 + returns).prod()
    
    # Annualize
    years = len(returns) / periods_per_year
    annualized_return = compound_return ** (1 / years) - 1
    
    return annualized_return


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize the volatility of a series of returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate Sharpe ratio
    if excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252, 
                target_return: float = 0.0) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        target_return: Minimum acceptable return
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
    
    # Calculate Sortino ratio
    return (excess_returns.mean() * periods_per_year) / downside_deviation


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate maximum drawdown
    max_dd = maximum_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if returns.mean() > 0 else 0.0
    
    # Calculate annualized return
    ann_return = annualize_return(returns, periods_per_year)
    
    # Calculate Calmar ratio
    return ann_return / abs(max_dd)


def omega_ratio(returns: pd.Series, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
    """
    Calculate the Omega ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        target_return: Target return threshold
        
    Returns:
        Omega ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - target_return
    
    # Calculate positive and negative returns
    positive_returns = excess_returns[excess_returns > 0]
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0 or abs(negative_returns.sum()) == 0:
        return np.inf if len(positive_returns) > 0 else 0.0
    
    # Calculate Omega ratio
    return positive_returns.sum() / abs(negative_returns.sum())


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Information ratio.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate tracking error
    tracking_error = (returns - benchmark_returns).std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    # Calculate Information ratio
    return ((returns - benchmark_returns).mean() * periods_per_year) / tracking_error


def maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a positive decimal
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns / running_max - 1)
    
    # Return the maximum drawdown as a positive number
    return abs(drawdown.min())


def drawdown_periods(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown periods.
    
    Args:
        returns: Series of returns
        
    Returns:
        DataFrame with drawdown periods
    """
    if len(returns) < 2:
        return pd.DataFrame(columns=["start", "end", "drawdown", "recovery", "underwater"])
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns / running_max - 1)
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    
    # Get start and end of each drawdown period
    starts = []
    ends = []
    drawdowns = []
    recoveries = []
    underwater = []
    
    in_drawdown = False
    start_idx = None
    max_drawdown = 0
    
    for i, (date, is_dd) in enumerate(is_drawdown.items()):
        if is_dd and not in_drawdown:
            # Start of drawdown period
            in_drawdown = True
            start_idx = date
            max_drawdown = 0
        
        if in_drawdown:
            # Update maximum drawdown
            max_drawdown = min(max_drawdown, drawdown[date])
        
        if not is_dd and in_drawdown:
            # End of drawdown period
            in_drawdown = False
            end_idx = date
            
            # Calculate recovery time
            recovery_time = (end_idx - start_idx).days
            
            # Calculate underwater time
            underwater_time = recovery_time
            
            starts.append(start_idx)
            ends.append(end_idx)
            drawdowns.append(max_drawdown)
            recoveries.append(recovery_time)
            underwater.append(underwater_time)
    
    # If still in drawdown at the end
    if in_drawdown:
        end_idx = returns.index[-1]
        recovery_time = np.nan  # No recovery yet
        underwater_time = (end_idx - start_idx).days
        
        starts.append(start_idx)
        ends.append(end_idx)
        drawdowns.append(max_drawdown)
        recoveries.append(recovery_time)
        underwater.append(underwater_time)
    
    # Create DataFrame
    drawdown_df = pd.DataFrame({
        "start": starts,
        "end": ends,
        "drawdown": drawdowns,
        "recovery": recoveries,
        "underwater": underwater
    })
    
    return drawdown_df


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = "historical") -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Series of returns
        confidence: Confidence level
        method: Method to calculate VaR ('historical', 'gaussian', 'cornish_fisher')
        
    Returns:
        Value at Risk
    """
    if len(returns) < 2:
        return 0.0
    
    if method == "historical":
        # Historical VaR
        return -np.percentile(returns, 100 * (1 - confidence))
    
    elif method == "gaussian":
        # Parametric Gaussian VaR
        z_score = stats.norm.ppf(1 - confidence)
        return -(returns.mean() + z_score * returns.std())
    
    elif method == "cornish_fisher":
        # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
        z_score = stats.norm.ppf(1 - confidence)
        s = stats.skew(returns)
        k = stats.kurtosis(returns)
        
        # Cornish-Fisher adjustment
        z_cf = z_score + (z_score**2 - 1) * s / 6 + (z_score**3 - 3 * z_score) * k / 24 - (2 * z_score**3 - 5 * z_score) * s**2 / 36
        
        return -(returns.mean() + z_cf * returns.std())
    
    else:
        logger.warning(f"Unknown VaR method: {method}, using 'historical' instead")
        return value_at_risk(returns, confidence, "historical")


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Series of returns
        confidence: Confidence level
        
    Returns:
        Conditional Value at Risk
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate VaR
    var = value_at_risk(returns, confidence, "historical")
    
    # Calculate CVaR
    return -returns[returns <= -var].mean()


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate portfolio beta.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        
    Returns:
        Portfolio beta
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate covariance
    cov = returns.cov(benchmark_returns)
    
    # Calculate benchmark variance
    benchmark_var = benchmark_returns.var()
    
    if benchmark_var == 0:
        return 0.0
    
    # Calculate beta
    return cov / benchmark_var


def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0, 
         periods_per_year: int = 252) -> float:
    """
    Calculate portfolio alpha.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        
    Returns:
        Portfolio alpha (annualized)
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate beta
    b = beta(returns, benchmark_returns)
    
    # Calculate alpha
    alpha_per_period = returns.mean() - rf_per_period - b * (benchmark_returns.mean() - rf_per_period)
    
    # Annualize alpha
    return alpha_per_period * periods_per_year


def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0,
                periods_per_year: int = 252) -> float:
    """
    Calculate the Treynor ratio.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        
    Returns:
        Treynor ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate beta
    b = beta(returns, benchmark_returns)
    
    if b == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns.mean() - rf_per_period
    
    # Calculate Treynor ratio
    return (excess_returns * periods_per_year) / b


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate tracking error.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Tracking error (annualized)
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate tracking error
    return (returns - benchmark_returns).std() * np.sqrt(periods_per_year)


def downside_risk(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate downside risk.
    
    Args:
        returns: Series of returns
        target_return: Target return threshold
        periods_per_year: Number of periods in a year
        
    Returns:
        Downside risk (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate downside returns
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return 0.0
    
    # Calculate downside risk
    return np.sqrt(np.mean((downside_returns - target_return) ** 2)) * np.sqrt(periods_per_year)


def upside_potential_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """
    Calculate upside potential ratio.
    
    Args:
        returns: Series of returns
        target_return: Target return threshold
        
    Returns:
        Upside potential ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate upside and downside returns
    upside_returns = returns[returns > target_return]
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf if len(upside_returns) > 0 else 0.0
    
    # Calculate upside potential and downside risk
    upside_potential = np.mean(upside_returns - target_return) if len(upside_returns) > 0 else 0
    downside_risk_val = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    if downside_risk_val == 0:
        return np.inf if upside_potential > 0 else 0.0
    
    # Calculate upside potential ratio
    return upside_potential / downside_risk_val


def win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate.
    
    Args:
        returns: Series of returns
        
    Returns:
        Win rate (percentage)
    """
    if len(returns) < 1:
        return 0.0
    
    # Calculate win rate
    wins = (returns > 0).sum()
    return wins / len(returns) * 100


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor.
    
    Args:
        returns: Series of returns
        
    Returns:
        Profit factor
    """
    if len(returns) < 1:
        return 0.0
    
    # Calculate profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns (optional)
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary of portfolio metrics
    """
    metrics = {}
    
    # Basic return metrics
    metrics["total_return"] = (1 + returns).prod() - 1
    metrics["annualized_return"] = annualize_return(returns, periods_per_year)
    metrics["annualized_volatility"] = annualize_volatility(returns, periods_per_year)
    
    # Risk-adjusted return metrics
    metrics["sharpe_ratio"] = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics["sortino_ratio"] = sortino_ratio(returns, risk_free_rate, periods_per_year)
    metrics["calmar_ratio"] = calmar_ratio(returns, periods_per_year)
    metrics["omega_ratio"] = omega_ratio(returns, risk_free_rate)
    
    # Drawdown metrics
    metrics["max_drawdown"] = maximum_drawdown(returns)
    
    # Risk metrics
    metrics["var_95"] = value_at_risk(returns, 0.95)
    metrics["cvar_95"] = conditional_value_at_risk(returns, 0.95)
    metrics["downside_risk"] = downside_risk(returns, 0.0, periods_per_year)
    
    # Trading metrics
    metrics["win_rate"] = win_rate(returns)
    metrics["profit_factor"] = profit_factor(returns)
    
    # Benchmark-relative metrics (if benchmark provided)
    if benchmark_returns is not None:
        # Align returns and benchmark returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) >= 2:
            metrics["beta"] = beta(aligned_returns, aligned_benchmark)
            metrics["alpha"] = alpha(aligned_returns, aligned_benchmark, risk_free_rate, periods_per_year)
            metrics["information_ratio"] = information_ratio(aligned_returns, aligned_benchmark, periods_per_year)
            metrics["tracking_error"] = tracking_error(aligned_returns, aligned_benchmark, periods_per_year)
            metrics["treynor_ratio"] = treynor_ratio(aligned_returns, aligned_benchmark, risk_free_rate, periods_per_year)
    
    return metrics


def calculate_rolling_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    window: int = 63,  # ~3 months of trading days
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling portfolio metrics.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns (optional)
        risk_free_rate: Risk-free rate (annualized)
        window: Rolling window size
        periods_per_year: Number of periods in a year
        
    Returns:
        DataFrame of rolling metrics
    """
    if len(returns) < window:
        return pd.DataFrame()
    
    # Initialize metrics DataFrame
    metrics_df = pd.DataFrame(index=returns.index[window-1:])
    
    # Calculate rolling metrics
    metrics_df["rolling_return"] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    metrics_df["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(periods_per_year)
    metrics_df["rolling_sharpe"] = returns.rolling(window).apply(lambda x: sharpe_ratio(x, risk_free_rate, periods_per_year))
    metrics_df["rolling_sortino"] = returns.rolling(window).apply(lambda x: sortino_ratio(x, risk_free_rate, periods_per_year))
    metrics_df["rolling_max_drawdown"] = returns.rolling(window).apply(maximum_drawdown)
    
    # Calculate rolling benchmark-relative metrics
    if benchmark_returns is not None:
        # Align returns and benchmark returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) >= window:
            # Calculate rolling beta
            rolling_beta = pd.Series(index=aligned_returns.index[window-1:])
            for i in range(window-1, len(aligned_returns)):
                rolling_beta.iloc[i-window+1] = beta(
                    aligned_returns.iloc[i-window+1:i+1],
                    aligned_benchmark.iloc[i-window+1:i+1]
                )
            metrics_df["rolling_beta"] = rolling_beta
            
            # Calculate rolling alpha
            rolling_alpha = pd.Series(index=aligned_returns.index[window-1:])
            for i in range(window-1, len(aligned_returns)):
                rolling_alpha.iloc[i-window+1] = alpha(
                    aligned_returns.iloc[i-window+1:i+1],
                    aligned_benchmark.iloc[i-window+1:i+1],
                    risk_free_rate,
                    periods_per_year
                )
            metrics_df["rolling_alpha"] = rolling_alpha
    
    return metrics_df


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns.
    
    Args:
        returns: Series of daily returns
        
    Returns:
        DataFrame of monthly returns
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning("Returns index is not a DatetimeIndex, cannot calculate monthly returns")
        return pd.DataFrame()
    
    # Resample to monthly returns
    monthly_returns = (1 + returns).resample('M').prod() - 1
    
    # Create a DataFrame with years as rows and months as columns
    monthly_table = pd.DataFrame(
        index=monthly_returns.index.year.unique(),
        columns=range(1, 13)
    )
    
    # Fill the table with monthly returns
    for date, ret in monthly_returns.items():
        monthly_table.loc[date.year, date.month] = ret
    
    # Add annual returns
    monthly_table["Annual"] = monthly_table.apply(
        lambda x: (1 + x.dropna()).prod() - 1 if len(x.dropna()) > 0 else np.nan,
        axis=1
    )
    
    return monthly_table


def calculate_drawdowns(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate top drawdowns.
    
    Args:
        returns: Series of returns
        top_n: Number of top drawdowns to return
        
    Returns:
        DataFrame of top drawdowns
    """
    # Get drawdown periods
    dd_periods = drawdown_periods(returns)
    
    if len(dd_periods) == 0:
        return pd.DataFrame(columns=["start", "end", "drawdown", "recovery", "underwater"])
    
    # Sort by drawdown (ascending, as drawdowns are negative)
    dd_periods = dd_periods.sort_values("drawdown").head(top_n)
    
    # Format drawdown as percentage
    dd_periods["drawdown"] = dd_periods["drawdown"] * 100
    
    return dd_periods


def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-based metrics.
    
    Args:
        trades_df: DataFrame of trades with columns: pnl, return, holding_period
        
    Returns:
        Dictionary of trade metrics
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "avg_winning_return": 0.0,
            "avg_losing_return": 0.0,
            "avg_holding_period": 0.0
        }
    
    # Calculate trade metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df["pnl"] > 0]
    losing_trades = trades_df[trades_df["pnl"] < 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
    
    total_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0.0
    total_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1e-9
    profit_factor = total_profit / total_loss
    
    avg_return = trades_df["return"].mean() if "return" in trades_df.columns else 0.0
    avg_winning_return = winning_trades["return"].mean() if "return" in trades_df.columns and len(winning_trades) > 0 else 0.0
    avg_losing_return = losing_trades["return"].mean() if "return" in trades_df.columns and len(losing_trades) > 0 else 0.0
    
    avg_holding_period = trades_df["holding_period"].mean() if "holding_period" in trades_df.columns else 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "avg_winning_return": avg_winning_return,
        "avg_losing_return": avg_losing_return,
        "avg_holding_period": avg_holding_period
    }
