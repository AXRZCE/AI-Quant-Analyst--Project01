"""
Train enhanced reinforcement learning models for trading.

This script provides functionality for training enhanced reinforcement learning models
for trading strategies with realistic constraints.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import RL modules
from src.rl.enhanced_env import EnhancedTradingEnv, EnhancedMultiAssetTradingEnv
from src.rl.sac import SAC, train_sac
from src.rl.evaluation import evaluate_strategy
from src.rl.rl_strategy import RLStrategy, RLStrategyFactory
from src.rl.hyperparameter_tuning import grid_search, walk_forward_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "rl_training.log"))
    ]
)
logger = logging.getLogger(__name__)


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


def create_environment(
    df: pd.DataFrame,
    feature_cols: List[str],
    price_col: str = "close",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    window_size: int = 1,
    multi_asset: bool = False,
    **kwargs
) -> Union[EnhancedTradingEnv, EnhancedMultiAssetTradingEnv]:
    """
    Create enhanced trading environment.
    
    Args:
        df: DataFrame with data
        feature_cols: List of feature column names
        price_col: Name of price column
        date_col: Name of date column
        symbol_col: Name of symbol column
        initial_capital: Initial capital
        transaction_cost: Transaction cost as a fraction of trade value
        window_size: Number of time steps to include in the observation
        multi_asset: Whether to use multi-asset environment
        **kwargs: Additional arguments for enhanced environment
        
    Returns:
        Enhanced trading environment
    """
    logger.info("Creating enhanced trading environment")
    
    if multi_asset:
        env = EnhancedMultiAssetTradingEnv(
            df=df,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            window_size=window_size,
            symbol_col=symbol_col,
            price_col=price_col,
            date_col=date_col,
            **kwargs
        )
    else:
        env = EnhancedTradingEnv(
            df=df,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            window_size=window_size,
            symbol_col=symbol_col,
            price_col=price_col,
            date_col=date_col,
            **kwargs
        )
    
    return env


def train_rl_model(
    env,
    algorithm: str = "sac",
    num_episodes: int = 1000,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train reinforcement learning model.
    
    Args:
        env: Trading environment
        algorithm: RL algorithm to use ("sac" or "ppo")
        num_episodes: Number of episodes to train for
        save_path: Path to save model
        **kwargs: Additional arguments for training
        
    Returns:
        Tuple of (agent, training_info)
    """
    logger.info(f"Training {algorithm.upper()} model for {num_episodes} episodes")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    
    if algorithm.lower() == "sac":
        if isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n
        
        # Train SAC agent
        agent, training_info = train_sac(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            num_episodes=num_episodes,
            save_path=save_path,
            **kwargs
        )
    
    elif algorithm.lower() == "ppo":
        # Import PPO if available
        try:
            from src.rl.ppo import train_ppo
            
            if isinstance(env.action_space, gym.spaces.Box):
                action_dim = env.action_space.shape[0]
            else:
                action_dim = env.action_space.n
            
            # Train PPO agent
            agent, training_info = train_ppo(
                env=env,
                state_dim=state_dim,
                action_dim=action_dim,
                num_episodes=num_episodes,
                save_path=save_path,
                **kwargs
            )
        except ImportError:
            logger.error("PPO implementation not found. Please implement src.rl.ppo.train_ppo.")
            raise
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return agent, training_info


def create_rl_strategy(
    agent,
    algorithm: str = "sac",
    feature_cols: List[str] = None,
    price_col: str = "close",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    normalize_state: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> RLStrategy:
    """
    Create RL strategy from trained agent.
    
    Args:
        agent: Trained RL agent
        algorithm: RL algorithm used ("sac" or "ppo")
        feature_cols: List of feature column names
        price_col: Name of price column
        date_col: Name of date column
        symbol_col: Name of symbol column
        normalize_state: Whether to normalize state
        device: Device to use for inference
        
    Returns:
        RL strategy
    """
    logger.info("Creating RL strategy")
    
    # Get state and action dimensions
    state_dim = agent.state_dim
    action_dim = agent.action_dim
    
    # Create strategy
    strategy = RLStrategyFactory.create_strategy(
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        feature_cols=feature_cols,
        price_col=price_col,
        date_col=date_col,
        symbol_col=symbol_col,
        normalize_state=normalize_state,
        device=device
    )
    
    # Set agent
    strategy.agent = agent
    
    return strategy


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train enhanced reinforcement learning models for trading")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True, help="Feature column names")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--date-col", type=str, default="timestamp", help="Date column name")
    parser.add_argument("--symbol-col", type=str, default="symbol", help="Symbol column name")
    
    # Environment arguments
    parser.add_argument("--initial-capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost as a fraction of trade value")
    parser.add_argument("--window-size", type=int, default=1, help="Number of time steps to include in the observation")
    parser.add_argument("--multi-asset", action="store_true", help="Whether to use multi-asset environment")
    
    # Enhanced environment arguments
    parser.add_argument("--market-impact-factor", type=float, default=0.1, help="Factor for market impact model")
    parser.add_argument("--max-position-size", type=float, default=1.0, help="Maximum position size as a fraction of capital")
    parser.add_argument("--stop-loss-pct", type=float, default=None, help="Stop loss percentage")
    parser.add_argument("--take-profit-pct", type=float, default=None, help="Take profit percentage")
    parser.add_argument("--slippage-std", type=float, default=0.0005, help="Standard deviation for slippage model")
    parser.add_argument("--market-hours-only", action="store_true", help="Whether to only trade during market hours")
    parser.add_argument("--max-drawdown-pct", type=float, default=None, help="Maximum allowed drawdown percentage")
    
    # Training arguments
    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"], help="RL algorithm to use")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="Temperature parameter for entropy")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of hidden layers")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Capacity of replay buffer")
    
    # Evaluation arguments
    parser.add_argument("--eval-interval", type=int, default=10, help="Interval between evaluations")
    parser.add_argument("--num-eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models/rl", help="Directory to save models and results")
    parser.add_argument("--model-name", type=str, default=None, help="Model name (default: {algorithm}_{timestamp})")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create logs directory
        os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
        
        # Generate model name if not provided
        if args.model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.model_name = f"{args.algorithm}_{timestamp}"
        
        # Create model directory
        model_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(model_dir, "args.json"), "w") as f:
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
            "market_impact_factor": args.market_impact_factor,
            "max_position_size": args.max_position_size,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "slippage_std": args.slippage_std,
            "market_hours_only": args.market_hours_only,
            "max_drawdown_pct": args.max_drawdown_pct
        }
        
        env = create_environment(
            df=train_df,
            feature_cols=args.feature_cols,
            price_col=args.price_col,
            date_col=args.date_col,
            symbol_col=args.symbol_col,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            window_size=args.window_size,
            multi_asset=args.multi_asset,
            **env_kwargs
        )
        
        # Train model
        agent, training_info = train_rl_model(
            env=env,
            algorithm=args.algorithm,
            num_episodes=args.num_episodes,
            save_path=model_dir,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            hidden_dim=args.hidden_dim,
            buffer_capacity=args.buffer_capacity,
            eval_interval=args.eval_interval,
            num_eval_episodes=args.num_eval_episodes
        )
        
        # Save training info
        with open(os.path.join(model_dir, "training_info.json"), "w") as f:
            # Convert numpy arrays to lists
            for key, value in training_info.items():
                if isinstance(value, np.ndarray):
                    training_info[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    training_info[key] = [v.tolist() for v in value]
            
            json.dump(training_info, f, indent=4)
        
        # Create RL strategy
        strategy = create_rl_strategy(
            agent=agent,
            algorithm=args.algorithm,
            feature_cols=args.feature_cols,
            price_col=args.price_col,
            date_col=args.date_col,
            symbol_col=args.symbol_col
        )
        
        # Evaluate on validation set
        val_results = evaluate_strategy(
            strategy=strategy,
            test_df=val_df,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage_std
        )
        
        # Save validation results
        with open(os.path.join(model_dir, "val_results.json"), "w") as f:
            # Convert numpy arrays to lists
            for key, value in val_results["metrics"].items():
                if isinstance(value, np.ndarray):
                    val_results["metrics"][key] = value.tolist()
            
            json.dump(val_results["metrics"], f, indent=4)
        
        # Evaluate on test set
        test_results = evaluate_strategy(
            strategy=strategy,
            test_df=test_df,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage_std
        )
        
        # Save test results
        with open(os.path.join(model_dir, "test_results.json"), "w") as f:
            # Convert numpy arrays to lists
            for key, value in test_results["metrics"].items():
                if isinstance(value, np.ndarray):
                    test_results["metrics"][key] = value.tolist()
            
            json.dump(test_results["metrics"], f, indent=4)
        
        # Print results
        print("\nTraining complete!")
        print(f"Model saved to: {model_dir}")
        print("\nValidation Results:")
        for key, value in val_results["metrics"].items():
            print(f"  {key}: {value}")
        
        print("\nTest Results:")
        for key, value in test_results["metrics"].items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
