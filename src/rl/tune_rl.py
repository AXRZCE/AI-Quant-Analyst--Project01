"""
Tune hyperparameters for reinforcement learning models for trading.

This script provides functionality for tuning hyperparameters of reinforcement learning
models for trading strategies.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RL modules
from src.rl.enhanced_env import EnhancedTradingEnv, EnhancedMultiAssetTradingEnv
from src.rl.sac import SAC
from src.rl.rl_strategy import RLStrategy, RLStrategyFactory
from src.rl.hyperparameter_tuning import grid_search, bayesian_optimization, walk_forward_validation


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        DataFrame with data
    """
    logger.info(f"Loading data from {data_path}")
    
    # Check file extension
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Convert timestamp to datetime if it's a string
    if "timestamp" in df.columns and isinstance(df["timestamp"].iloc[0], str):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    price_col: str = "close",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for training, validation, and testing.
    
    Args:
        df: DataFrame with data
        feature_cols: List of feature column names
        price_col: Name of price column
        date_col: Name of date column
        symbol_col: Name of symbol column
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Preparing data")
    
    # Check ratios
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError(f"Ratios must sum to 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
    
    # Get unique dates
    dates = df[date_col].unique()
    dates.sort()
    
    # Calculate split indices
    train_end = int(len(dates) * train_ratio)
    val_end = train_end + int(len(dates) * val_ratio)
    
    # Split dates
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    # Split data
    train_df = df[df[date_col].isin(train_dates)]
    val_df = df[df[date_col].isin(val_dates)]
    test_df = df[df[date_col].isin(test_dates)]
    
    logger.info(f"Train: {len(train_df)} rows ({train_dates[0]} to {train_dates[-1]})")
    logger.info(f"Validation: {len(val_df)} rows ({val_dates[0]} to {val_dates[-1]})")
    logger.info(f"Test: {len(test_df)} rows ({test_dates[0]} to {test_dates[-1]})")
    
    return train_df, val_df, test_df


def create_strategy_class(
    algorithm: str = "sac",
    feature_cols: List[str] = None,
    price_col: str = "close",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    device: str = "cpu"
):
    """
    Create a strategy class for hyperparameter tuning.
    
    Args:
        algorithm: RL algorithm to use
        feature_cols: Feature column names
        price_col: Price column name
        date_col: Date column name
        symbol_col: Symbol column name
        device: Device to use for inference
        
    Returns:
        Strategy class
    """
    class TunableStrategy:
        """Tunable strategy for hyperparameter tuning."""
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 0.2,
            batch_size: int = 256,
            buffer_capacity: int = 1_000_000,
            num_episodes: int = 100,
            **kwargs
        ):
            """
            Initialize tunable strategy.
            
            Args:
                state_dim: State dimension
                action_dim: Action dimension
                hidden_dim: Hidden dimension
                learning_rate: Learning rate
                gamma: Discount factor
                tau: Target network update rate
                alpha: Temperature parameter for entropy
                batch_size: Batch size
                buffer_capacity: Buffer capacity
                num_episodes: Number of episodes
                **kwargs: Additional arguments
            """
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.tau = tau
            self.alpha = alpha
            self.batch_size = batch_size
            self.buffer_capacity = buffer_capacity
            self.num_episodes = num_episodes
            self.kwargs = kwargs
            
            # Initialize agent
            if algorithm.lower() == "sac":
                self.agent = SAC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    lr=learning_rate,
                    gamma=gamma,
                    tau=tau,
                    alpha=alpha,
                    batch_size=batch_size,
                    buffer_capacity=buffer_capacity,
                    device=device
                )
            elif algorithm.lower() == "ppo":
                # Import PPO if available
                try:
                    from src.rl.ppo import PPO
                    
                    self.agent = PPO(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=hidden_dim,
                        lr=learning_rate,
                        gamma=gamma,
                        device=device
                    )
                except ImportError:
                    logger.error("PPO implementation not found. Please implement src.rl.ppo.PPO.")
                    raise
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Initialize strategy
            self.strategy = RLStrategyFactory.create_strategy(
                algorithm=algorithm,
                state_dim=state_dim,
                action_dim=action_dim,
                feature_cols=feature_cols,
                price_col=price_col,
                date_col=date_col,
                symbol_col=symbol_col,
                device=device
            )
            
            # Set agent
            self.strategy.agent = self.agent
        
        def train(self, train_df: pd.DataFrame) -> None:
            """
            Train strategy.
            
            Args:
                train_df: Training data
            """
            # Create environment
            if "multi_asset" in self.kwargs and self.kwargs["multi_asset"]:
                env = EnhancedMultiAssetTradingEnv(
                    df=train_df,
                    feature_cols=feature_cols,
                    price_col=price_col,
                    date_col=date_col,
                    symbol_col=symbol_col,
                    **{k: v for k, v in self.kwargs.items() if k != "multi_asset"}
                )
            else:
                env = EnhancedTradingEnv(
                    df=train_df,
                    feature_cols=feature_cols,
                    price_col=price_col,
                    date_col=date_col,
                    symbol_col=symbol_col,
                    **{k: v for k, v in self.kwargs.items() if k != "multi_asset"}
                )
            
            # Train agent
            if algorithm.lower() == "sac":
                self.agent.train(
                    env=env,
                    num_episodes=self.num_episodes,
                    batch_size=self.batch_size,
                    verbose=False
                )
            elif algorithm.lower() == "ppo":
                self.agent.train(
                    env=env,
                    num_episodes=self.num_episodes,
                    verbose=False
                )
        
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
                df: DataFrame with data
                initial_capital: Initial capital
                transaction_cost: Transaction cost as a fraction of trade value
                slippage: Slippage as a fraction of price
                
            Returns:
                Dictionary with backtest results
            """
            return self.strategy.backtest(
                df=df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage
            )
    
    return TunableStrategy


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for reinforcement learning models for trading")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True, help="Feature column names")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--date-col", type=str, default="timestamp", help="Date column name")
    parser.add_argument("--symbol-col", type=str, default="symbol", help="Symbol column name")
    
    # Environment arguments
    parser.add_argument("--initial-capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost as a fraction of trade value")
    parser.add_argument("--multi-asset", action="store_true", help="Whether to use multi-asset environment")
    
    # Enhanced environment arguments
    parser.add_argument("--market-impact-factor", type=float, default=0.1, help="Factor for market impact model")
    parser.add_argument("--max-position-size", type=float, default=1.0, help="Maximum position size as a fraction of capital")
    parser.add_argument("--stop-loss-pct", type=float, default=None, help="Stop loss percentage")
    parser.add_argument("--take-profit-pct", type=float, default=None, help="Take profit percentage")
    parser.add_argument("--slippage-std", type=float, default=0.0005, help="Standard deviation for slippage model")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"], help="RL algorithm to use")
    
    # Tuning arguments
    parser.add_argument("--tuning-method", type=str, default="grid", choices=["grid", "bayesian", "walk_forward"], help="Tuning method")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric to optimize")
    parser.add_argument("--maximize", action="store_true", help="Whether to maximize the metric")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of trials for random search")
    parser.add_argument("--random-search", action="store_true", help="Whether to use random search instead of grid search")
    
    # Grid search arguments
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3], help="Learning rates to search")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 128, 256], help="Hidden dimensions to search")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.9, 0.95, 0.99], help="Discount factors to search")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.001, 0.005, 0.01], help="Target network update rates to search")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="Temperature parameters to search")
    
    # Training arguments
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes for training")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models/rl_tuning", help="Directory to save models and results")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Check if CUDA is available
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU instead.")
            args.device = "cpu"
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        
        # Load data
        df = load_data(args.data_path)
        
        # Prepare data
        train_df, val_df, test_df = prepare_data(
            df=df,
            feature_cols=args.feature_cols,
            price_col=args.price_col,
            date_col=args.date_col,
            symbol_col=args.symbol_col
        )
        
        # Create environment
        env_kwargs = {
            "initial_capital": args.initial_capital,
            "transaction_cost": args.transaction_cost,
            "multi_asset": args.multi_asset,
            "market_impact_factor": args.market_impact_factor,
            "max_position_size": args.max_position_size,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "slippage_std": args.slippage_std
        }
        
        # Create strategy class
        strategy_class = create_strategy_class(
            algorithm=args.algorithm,
            feature_cols=args.feature_cols,
            price_col=args.price_col,
            date_col=args.date_col,
            symbol_col=args.symbol_col,
            device=args.device
        )
        
        # Create parameter grid
        param_grid = {
            "state_dim": [len(args.feature_cols)],
            "action_dim": [1 if not args.multi_asset else len(df[args.symbol_col].unique())],
            "hidden_dim": args.hidden_dims,
            "learning_rate": args.learning_rates,
            "gamma": args.gammas,
            "tau": args.taus,
            "alpha": args.alphas,
            "batch_size": [args.batch_size],
            "buffer_capacity": [1_000_000],
            "num_episodes": [args.num_episodes],
            **env_kwargs
        }
        
        # Run hyperparameter tuning
        if args.tuning_method == "grid":
            # Run grid search
            tuning_results = grid_search(
                strategy_class=strategy_class,
                param_grid=param_grid,
                train_df=train_df,
                val_df=val_df,
                metric=args.metric,
                maximize=args.maximize,
                initial_capital=args.initial_capital,
                transaction_cost=args.transaction_cost,
                slippage=args.slippage_std,
                n_trials=args.n_trials,
                random_search=args.random_search,
                verbose=True
            )
        elif args.tuning_method == "bayesian":
            # Create parameter space for Bayesian optimization
            param_space = {
                "state_dim": [len(args.feature_cols)],
                "action_dim": [1 if not args.multi_asset else len(df[args.symbol_col].unique())],
                "hidden_dim": (64, 256, "int"),
                "learning_rate": (1e-4, 1e-3, "float"),
                "gamma": (0.9, 0.99, "float"),
                "tau": (0.001, 0.01, "float"),
                "alpha": (0.1, 0.3, "float"),
                "batch_size": [args.batch_size],
                "buffer_capacity": [1_000_000],
                "num_episodes": [args.num_episodes],
                **env_kwargs
            }
            
            # Run Bayesian optimization
            tuning_results = bayesian_optimization(
                strategy_class=strategy_class,
                param_space=param_space,
                train_df=train_df,
                val_df=val_df,
                metric=args.metric,
                maximize=args.maximize,
                initial_capital=args.initial_capital,
                transaction_cost=args.transaction_cost,
                slippage=args.slippage_std,
                n_trials=args.n_trials or 20,
                verbose=True
            )
        elif args.tuning_method == "walk_forward":
            # Use best parameters from grid search
            best_params = {
                "state_dim": len(args.feature_cols),
                "action_dim": 1 if not args.multi_asset else len(df[args.symbol_col].unique()),
                "hidden_dim": args.hidden_dims[0],
                "learning_rate": args.learning_rates[0],
                "gamma": args.gammas[0],
                "tau": args.taus[0],
                "alpha": args.alphas[0],
                "batch_size": args.batch_size,
                "buffer_capacity": 1_000_000,
                "num_episodes": args.num_episodes,
                **env_kwargs
            }
            
            # Run walk-forward validation
            tuning_results = walk_forward_validation(
                strategy_class=strategy_class,
                params=best_params,
                df=df,
                train_window=252,  # 1 year of trading days
                val_window=63,     # 3 months of trading days
                step=21,           # 1 month of trading days
                metric=args.metric,
                initial_capital=args.initial_capital,
                transaction_cost=args.transaction_cost,
                slippage=args.slippage_std,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported tuning method: {args.tuning_method}")
        
        # Save tuning results
        with open(os.path.join(args.output_dir, "tuning_results.json"), "w") as f:
            # Convert numpy arrays to lists
            if args.tuning_method == "grid" or args.tuning_method == "bayesian":
                # Save best parameters
                best_params = tuning_results["best_params"]
                
                # Convert numpy values to Python types
                for key, value in best_params.items():
                    if isinstance(value, np.ndarray):
                        best_params[key] = value.tolist()
                    elif isinstance(value, np.float64):
                        best_params[key] = float(value)
                    elif isinstance(value, np.int64):
                        best_params[key] = int(value)
                
                # Save results
                results_to_save = {
                    "best_params": best_params,
                    "best_score": float(tuning_results["best_score"]),
                    "results": []
                }
                
                # Convert results
                for result in tuning_results["results"]:
                    result_dict = {}
                    
                    # Convert params
                    params = result["params"]
                    for key, value in params.items():
                        if isinstance(value, np.ndarray):
                            params[key] = value.tolist()
                        elif isinstance(value, np.float64):
                            params[key] = float(value)
                        elif isinstance(value, np.int64):
                            params[key] = int(value)
                    
                    result_dict["params"] = params
                    
                    # Convert score
                    result_dict["score"] = float(result["score"])
                    
                    # Convert metrics
                    if "metrics" in result and result["metrics"] is not None:
                        metrics = result["metrics"]
                        for key, value in metrics.items():
                            if isinstance(value, np.ndarray):
                                metrics[key] = value.tolist()
                            elif isinstance(value, np.float64):
                                metrics[key] = float(value)
                            elif isinstance(value, np.int64):
                                metrics[key] = int(value)
                        
                        result_dict["metrics"] = metrics
                    
                    # Add to results
                    results_to_save["results"].append(result_dict)
            else:
                # Save walk-forward results
                results_to_save = {
                    "params": tuning_results["params"],
                    "mean_score": float(tuning_results["mean_score"]),
                    "std_score": float(tuning_results["std_score"]),
                    "results": []
                }
                
                # Convert results
                for result in tuning_results["results"]:
                    result_dict = {}
                    
                    # Convert values
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            result_dict[key] = value.tolist()
                        elif isinstance(value, np.float64):
                            result_dict[key] = float(value)
                        elif isinstance(value, np.int64):
                            result_dict[key] = int(value)
                        else:
                            result_dict[key] = value
                    
                    # Add to results
                    results_to_save["results"].append(result_dict)
            
            # Save to file
            json.dump(results_to_save, f, indent=4)
        
        # Print results
        print("\nTuning complete!")
        print(f"Results saved to: {os.path.join(args.output_dir, 'tuning_results.json')}")
        
        if args.tuning_method == "grid" or args.tuning_method == "bayesian":
            print("\nBest Parameters:")
            for key, value in tuning_results["best_params"].items():
                print(f"  {key}: {value}")
            
            print(f"\nBest Score ({args.metric}): {tuning_results['best_score']}")
        else:
            print("\nWalk-Forward Validation Results:")
            print(f"  Mean Score ({args.metric}): {tuning_results['mean_score']}")
            print(f"  Std Score ({args.metric}): {tuning_results['std_score']}")
        
        # Train final model with best parameters
        if args.tuning_method == "grid" or args.tuning_method == "bayesian":
            # Get best parameters
            best_params = tuning_results["best_params"]
            
            # Create strategy
            strategy = strategy_class(**best_params)
            
            # Train on full dataset
            strategy.train(df)
            
            # Save model
            model_path = os.path.join(args.output_dir, "best_model.pt")
            strategy.agent.save(model_path)
            
            print(f"\nFinal model saved to: {model_path}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
