"""
Monte Carlo simulation for scenario generation.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from numba import njit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@njit
def simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion.
    
    Args:
        S0: Initial price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        T: Time horizon in days
        steps: Number of steps
        n_paths: Number of paths to simulate
        
    Returns:
        Array of simulated paths with shape (n_paths, steps + 1)
    """
    dt = T / steps
    paths = np.empty((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, steps + 1):
        z = np.random.normal(size=n_paths)
        paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

def run_simulation(
    S0: float,
    mu: float,
    sigma: float,
    days: float = 1.0,
    steps_per_day: int = 390,
    n_paths: int = 1000
) -> np.ndarray:
    """
    Run Monte Carlo simulation.
    
    Args:
        S0: Initial price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        days: Time horizon in days
        steps_per_day: Number of steps per day
        n_paths: Number of paths to simulate
        
    Returns:
        Array of simulated paths with shape (n_paths, steps_per_day * days + 1)
    """
    logger.info(f"Running Monte Carlo simulation with {n_paths} paths")
    
    # Convert daily parameters to annual
    mu_annual = mu * 252
    sigma_annual = sigma * np.sqrt(252)
    
    # Run simulation
    steps = int(steps_per_day * days)
    paths = simulate_paths(S0, mu_annual, sigma_annual, days / 252, steps, n_paths)
    
    logger.info(f"Simulation completed with shape {paths.shape}")
    
    return paths

@njit
def simulate_correlated_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    T: float,
    steps: int,
    n_paths: int
) -> np.ndarray:
    """
    Simulate correlated price paths using Geometric Brownian Motion.
    
    Args:
        S0: Initial prices (array of shape (n_assets,))
        mu: Drifts (annualized, array of shape (n_assets,))
        sigma: Volatilities (annualized, array of shape (n_assets,))
        corr_matrix: Correlation matrix (array of shape (n_assets, n_assets))
        T: Time horizon in days
        steps: Number of steps
        n_paths: Number of paths to simulate
        
    Returns:
        Array of simulated paths with shape (n_assets, n_paths, steps + 1)
    """
    n_assets = len(S0)
    dt = T / steps
    
    # Compute Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    
    # Initialize paths
    paths = np.empty((n_assets, n_paths, steps + 1))
    for i in range(n_assets):
        paths[i, :, 0] = S0[i]
    
    # Simulate paths
    for t in range(1, steps + 1):
        # Generate correlated normal random variables
        z = np.random.normal(size=(n_paths, n_assets))
        correlated_z = z @ L.T
        
        # Update paths
        for i in range(n_assets):
            paths[i, :, t] = paths[i, :, t-1] * np.exp(
                (mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * np.sqrt(dt) * correlated_z[:, i]
            )
    
    return paths

def run_correlated_simulation(
    S0: Union[List[float], np.ndarray],
    mu: Union[List[float], np.ndarray],
    sigma: Union[List[float], np.ndarray],
    corr_matrix: Optional[np.ndarray] = None,
    days: float = 1.0,
    steps_per_day: int = 390,
    n_paths: int = 1000
) -> np.ndarray:
    """
    Run Monte Carlo simulation for multiple correlated assets.
    
    Args:
        S0: Initial prices (list or array of shape (n_assets,))
        mu: Drifts (annualized, list or array of shape (n_assets,))
        sigma: Volatilities (annualized, list or array of shape (n_assets,))
        corr_matrix: Correlation matrix (array of shape (n_assets, n_assets))
        days: Time horizon in days
        steps_per_day: Number of steps per day
        n_paths: Number of paths to simulate
        
    Returns:
        Array of simulated paths with shape (n_assets, n_paths, steps_per_day * days + 1)
    """
    logger.info(f"Running correlated Monte Carlo simulation with {n_paths} paths")
    
    # Convert to numpy arrays
    S0 = np.array(S0)
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    n_assets = len(S0)
    
    # Create default correlation matrix if not provided
    if corr_matrix is None:
        corr_matrix = np.eye(n_assets)
    
    # Convert daily parameters to annual
    mu_annual = mu * 252
    sigma_annual = sigma * np.sqrt(252)
    
    # Run simulation
    steps = int(steps_per_day * days)
    paths = simulate_correlated_paths(
        S0, mu_annual, sigma_annual, corr_matrix, days / 252, steps, n_paths
    )
    
    logger.info(f"Simulation completed with shape {paths.shape}")
    
    return paths

def calculate_scenario_statistics(
    paths: np.ndarray,
    quantiles: Optional[List[float]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate statistics from simulated paths.
    
    Args:
        paths: Array of simulated paths with shape (n_paths, steps + 1)
        quantiles: List of quantiles to calculate
        
    Returns:
        Dictionary with statistics
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    # Calculate statistics
    mean_path = np.mean(paths, axis=0)
    std_path = np.std(paths, axis=0)
    quantile_paths = np.quantile(paths, quantiles, axis=0)
    
    # Calculate returns
    returns = paths[:, -1] / paths[:, 0] - 1
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    quantile_returns = np.quantile(returns, quantiles)
    
    # Calculate value at risk
    var_95 = -np.percentile(returns, 5)
    var_99 = -np.percentile(returns, 1)
    
    # Calculate expected shortfall
    es_95 = -np.mean(returns[returns <= -var_95])
    es_99 = -np.mean(returns[returns <= -var_99])
    
    return {
        "mean_path": mean_path,
        "std_path": std_path,
        "quantile_paths": quantile_paths,
        "mean_return": mean_return,
        "std_return": std_return,
        "quantile_returns": quantile_returns,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99
    }

def plot_scenarios(
    paths: np.ndarray,
    n_samples: int = 100,
    quantiles: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Monte Carlo Simulation",
    xlabel: str = "Time Steps",
    ylabel: str = "Price"
) -> plt.Figure:
    """
    Plot Monte Carlo scenarios.
    
    Args:
        paths: Array of simulated paths with shape (n_paths, steps + 1)
        n_samples: Number of sample paths to plot
        quantiles: List of quantiles to plot
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    # Calculate statistics
    stats = calculate_scenario_statistics(paths, quantiles)
    mean_path = stats["mean_path"]
    quantile_paths = stats["quantile_paths"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot sample paths
    indices = np.random.choice(paths.shape[0], min(n_samples, paths.shape[0]), replace=False)
    for i in indices:
        ax.plot(paths[i], color="lightgray", alpha=0.3)
    
    # Plot quantiles
    for i, q in enumerate(quantiles):
        ax.plot(quantile_paths[i], label=f"{q:.0%} Quantile", linestyle="--")
    
    # Plot mean path
    ax.plot(mean_path, label="Mean", color="black", linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    return fig

def plot_return_distribution(
    paths: np.ndarray,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Return Distribution",
    xlabel: str = "Return",
    ylabel: str = "Frequency"
) -> plt.Figure:
    """
    Plot the distribution of returns.
    
    Args:
        paths: Array of simulated paths with shape (n_paths, steps + 1)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    # Calculate returns
    returns = paths[:, -1] / paths[:, 0] - 1
    
    # Calculate statistics
    stats = calculate_scenario_statistics(paths)
    mean_return = stats["mean_return"]
    var_95 = stats["var_95"]
    var_99 = stats["var_99"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(returns, bins=50, density=True, alpha=0.7)
    
    # Plot mean
    ax.axvline(mean_return, color="black", linestyle="-", label=f"Mean: {mean_return:.2%}")
    
    # Plot VaR
    ax.axvline(-var_95, color="red", linestyle="--", label=f"95% VaR: {var_95:.2%}")
    ax.axvline(-var_99, color="darkred", linestyle="--", label=f"99% VaR: {var_99:.2%}")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    return fig

if __name__ == "__main__":
    # Example usage
    S0 = 100.0
    mu = 0.0005  # Daily drift
    sigma = 0.02  # Daily volatility
    
    # Run simulation
    paths = run_simulation(S0, mu, sigma, days=5, steps_per_day=10, n_paths=1000)
    
    # Calculate statistics
    stats = calculate_scenario_statistics(paths)
    
    print(f"Mean return: {stats['mean_return']:.2%}")
    print(f"Standard deviation: {stats['std_return']:.2%}")
    print(f"95% VaR: {stats['var_95']:.2%}")
    print(f"99% VaR: {stats['var_99']:.2%}")
    print(f"95% ES: {stats['es_95']:.2%}")
    print(f"99% ES: {stats['es_99']:.2%}")
    
    # Plot scenarios
    fig1 = plot_scenarios(paths)
    
    # Plot return distribution
    fig2 = plot_return_distribution(paths)
    
    # Save figures
    os.makedirs("figures", exist_ok=True)
    fig1.savefig("figures/monte_carlo_scenarios.png")
    fig2.savefig("figures/return_distribution.png")
    
    print("Figures saved to figures/ directory")
