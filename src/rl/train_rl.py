"""
Train a reinforcement learning agent for trading.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ray, but don't fail if it's not available
try:
    import ray
    from ray import tune
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray is not available. Please install it with 'pip install ray[rllib]'.")
    RAY_AVAILABLE = False

# Import local modules
from src.rl.env import TradingEnv

def env_creator(cfg: Dict[str, Any]) -> TradingEnv:
    """
    Create a trading environment from configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Trading environment
    """
    # Load data
    data_path = cfg.get("data_path")
    if not data_path:
        raise ValueError("data_path must be specified in the configuration")
    
    # Check if data_path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Get feature columns
    feature_cols = cfg.get("feature_cols")
    if not feature_cols:
        raise ValueError("feature_cols must be specified in the configuration")
    
    # Create environment
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        initial_capital=cfg.get("initial_capital", 100_000),
        transaction_cost=cfg.get("transaction_cost", 0.001),
        reward_scaling=cfg.get("reward_scaling", 1.0),
        window_size=cfg.get("window_size", 1),
        symbol_col=cfg.get("symbol_col", "symbol"),
        price_col=cfg.get("price_col", "close"),
        date_col=cfg.get("date_col", "timestamp")
    )
    
    return env

def train_ppo(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Train a PPO agent for trading.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Training results
    """
    if not RAY_AVAILABLE:
        logger.error("Ray is not available. Please install it with 'pip install ray[rllib]'.")
        return {"status": "error", "message": "Ray is not available"}
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, local_mode=args.debug)
    
    # Register environment
    register_env("trading_env", lambda cfg: env_creator(cfg))
    
    # Create environment config
    env_config = {
        "data_path": args.data_path,
        "feature_cols": args.feature_cols,
        "initial_capital": args.initial_capital,
        "transaction_cost": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "window_size": args.window_size,
        "symbol_col": args.symbol_col,
        "price_col": args.price_col,
        "date_col": args.date_col
    }
    
    # Create tune config
    tune_config = {
        "env": "trading_env",
        "env_config": env_config,
        "framework": "torch",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "model": {
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
        },
        "lr": tune.grid_search(args.learning_rates) if args.learning_rates else 1e-4,
        "train_batch_size": args.batch_size,
        "rollout_fragment_length": args.rollout_length,
        "sgd_minibatch_size": min(args.batch_size, 128),
        "num_sgd_iter": 10,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
    }
    
    # Run training
    logger.info("Starting training...")
    analysis = tune.run(
        PPOTrainer,
        name=args.experiment_name,
        config=tune_config,
        stop={"training_iteration": args.max_iters},
        checkpoint_at_end=True,
        local_dir=args.log_dir,
        verbose=args.verbose
    )
    
    # Get best config
    best_config = analysis.get_best_config(metric="episode_reward_mean")
    logger.info(f"Best config: {best_config}")
    
    # Get best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean"
    )
    logger.info(f"Best checkpoint: {best_checkpoint}")
    
    return {
        "status": "success",
        "best_config": best_config,
        "best_checkpoint": best_checkpoint,
        "analysis": analysis
    }

def train_sac(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Train a SAC agent for trading.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Training results
    """
    if not RAY_AVAILABLE:
        logger.error("Ray is not available. Please install it with 'pip install ray[rllib]'.")
        return {"status": "error", "message": "Ray is not available"}
    
    # Import SAC
    try:
        from ray.rllib.agents.sac import SACTrainer
    except ImportError:
        logger.error("SAC is not available in your Ray installation.")
        return {"status": "error", "message": "SAC is not available"}
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, local_mode=args.debug)
    
    # Register environment
    register_env("trading_env", lambda cfg: env_creator(cfg))
    
    # Create environment config
    env_config = {
        "data_path": args.data_path,
        "feature_cols": args.feature_cols,
        "initial_capital": args.initial_capital,
        "transaction_cost": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "window_size": args.window_size,
        "symbol_col": args.symbol_col,
        "price_col": args.price_col,
        "date_col": args.date_col
    }
    
    # Create tune config
    tune_config = {
        "env": "trading_env",
        "env_config": env_config,
        "framework": "torch",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": tune.grid_search(args.learning_rates) if args.learning_rates else 3e-4,
        "train_batch_size": args.batch_size,
        "rollout_fragment_length": args.rollout_length,
        "target_network_update_freq": 1,
        "tau": 0.005,
        "initial_alpha": 1.0,
        "normalize_actions": False,
        "no_done_at_end": True,
        "optimization": {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
    }
    
    # Run training
    logger.info("Starting training...")
    analysis = tune.run(
        SACTrainer,
        name=args.experiment_name,
        config=tune_config,
        stop={"training_iteration": args.max_iters},
        checkpoint_at_end=True,
        local_dir=args.log_dir,
        verbose=args.verbose
    )
    
    # Get best config
    best_config = analysis.get_best_config(metric="episode_reward_mean")
    logger.info(f"Best config: {best_config}")
    
    # Get best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean"
    )
    logger.info(f"Best checkpoint: {best_checkpoint}")
    
    return {
        "status": "success",
        "best_config": best_config,
        "best_checkpoint": best_checkpoint,
        "analysis": analysis
    }

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train agent
    if args.algorithm.lower() == "ppo":
        result = train_ppo(args)
    elif args.algorithm.lower() == "sac":
        result = train_sac(args)
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return
    
    # Check result
    if result["status"] == "error":
        logger.error(f"Training failed: {result['message']}")
        return
    
    # Print best checkpoint
    logger.info(f"Training completed successfully.")
    logger.info(f"Best checkpoint: {result['best_checkpoint']}")
    
    # Save best checkpoint path to file
    checkpoint_file = os.path.join(args.log_dir, f"{args.experiment_name}_best_checkpoint.txt")
    with open(checkpoint_file, "w") as f:
        f.write(result["best_checkpoint"])
    logger.info(f"Best checkpoint path saved to {checkpoint_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent for trading.")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True, help="Feature column names")
    parser.add_argument("--symbol-col", type=str, default="symbol", help="Symbol column name")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--date-col", type=str, default="timestamp", help="Date column name")
    
    # Environment arguments
    parser.add_argument("--initial-capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost")
    parser.add_argument("--reward-scaling", type=float, default=1.0, help="Reward scaling factor")
    parser.add_argument("--window-size", type=int, default=1, help="Window size for observation")
    
    # Training arguments
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm to use")
    parser.add_argument("--experiment-name", type=str, default="RL_Trading", help="Experiment name")
    parser.add_argument("--log-dir", type=str, default="logs/rl", help="Log directory")
    parser.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs to use")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--max-iters", type=int, default=50, help="Maximum number of training iterations")
    parser.add_argument("--batch-size", type=int, default=2000, help="Training batch size")
    parser.add_argument("--rollout-length", type=int, default=200, help="Rollout fragment length")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=None, help="Learning rates to search")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    main(args)
