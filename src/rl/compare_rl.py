"""
Compare different reinforcement learning algorithms for trading.

This script provides functionality for comparing different reinforcement learning
algorithms for trading strategies.
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
from src.rl.sac import SAC
from src.rl.rl_strategy import RLStrategy, RLStrategyFactory
from src.rl.evaluation import evaluate_strategy, compare_strategies, plot_comparison


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


def load_model(model_path: str, algorithm: str = "sac", device: str = "cpu") -> Any:
    """
    Load trained RL model.
    
    Args:
        model_path: Path to model file
        algorithm: RL algorithm used ("sac" or "ppo")
        device: Device to use for inference
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {algorithm.upper()} model from {model_path}")
    
    if algorithm.lower() == "sac":
        # Load model configuration
        config_path = os.path.join(os.path.dirname(model_path), "args.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Get state and action dimensions
            state_dim = config.get("state_dim")
            action_dim = config.get("action_dim")
            hidden_dim = config.get("hidden_dim", 256)
            
            if state_dim is None or action_dim is None:
                raise ValueError("state_dim and action_dim must be specified in the configuration")
        else:
            # Use default values
            logger.warning(f"Configuration file not found: {config_path}")
            logger.warning("Using default values for state_dim and action_dim")
            state_dim = 10  # Default value
            action_dim = 1  # Default value
            hidden_dim = 256  # Default value
        
        # Create model
        model = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # Load model weights
        model.load(model_path)
    
    elif algorithm.lower() == "ppo":
        # Import PPO if available
        try:
            from src.rl.ppo import PPO
            
            # Load model configuration
            config_path = os.path.join(os.path.dirname(model_path), "args.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Get state and action dimensions
                state_dim = config.get("state_dim")
                action_dim = config.get("action_dim")
                hidden_dim = config.get("hidden_dim", 256)
                
                if state_dim is None or action_dim is None:
                    raise ValueError("state_dim and action_dim must be specified in the configuration")
            else:
                # Use default values
                logger.warning(f"Configuration file not found: {config_path}")
                logger.warning("Using default values for state_dim and action_dim")
                state_dim = 10  # Default value
                action_dim = 1  # Default value
                hidden_dim = 256  # Default value
            
            # Create model
            model = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device
            )
            
            # Load model weights
            model.load(model_path)
        except ImportError:
            logger.error("PPO implementation not found. Please implement src.rl.ppo.PPO.")
            raise
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return model


def create_strategy(
    model,
    algorithm: str,
    feature_cols: List[str],
    price_col: str = "close",
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    device: str = "cpu"
) -> RLStrategy:
    """
    Create RL strategy from model.
    
    Args:
        model: Trained model
        algorithm: RL algorithm used
        feature_cols: Feature column names
        price_col: Price column name
        date_col: Date column name
        symbol_col: Symbol column name
        device: Device to use for inference
        
    Returns:
        RL strategy
    """
    # Create strategy
    strategy = RLStrategyFactory.create_strategy(
        algorithm=algorithm,
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        feature_cols=feature_cols,
        price_col=price_col,
        date_col=date_col,
        symbol_col=symbol_col,
        device=device
    )
    
    # Set model
    strategy.agent = model
    
    return strategy


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare different reinforcement learning algorithms for trading")
    
    # Model arguments
    parser.add_argument("--model-paths", type=str, nargs="+", required=True, help="Paths to model files")
    parser.add_argument("--model-names", type=str, nargs="+", help="Names of models (default: algorithm_index)")
    parser.add_argument("--algorithms", type=str, nargs="+", help="RL algorithms used (default: sac for all)")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True, help="Feature column names")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--date-col", type=str, default="timestamp", help="Date column name")
    parser.add_argument("--symbol-col", type=str, default="symbol", help="Symbol column name")
    
    # Evaluation arguments
    parser.add_argument("--initial-capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost as a fraction of trade value")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage as a fraction of price")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Risk-free rate (annualized)")
    
    # Output arguments
    parser.add_argument("--output-file", type=str, default=None, help="Path to output file")
    parser.add_argument("--plot", action="store_true", help="Whether to plot comparison")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Check if CUDA is available
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU instead.")
            args.device = "cpu"
        
        # Check if model paths and names have the same length
        if args.model_names and len(args.model_paths) != len(args.model_names):
            raise ValueError("Number of model paths and names must be the same")
        
        # Check if model paths and algorithms have the same length
        if args.algorithms and len(args.model_paths) != len(args.algorithms):
            raise ValueError("Number of model paths and algorithms must be the same")
        
        # Generate model names if not provided
        if not args.model_names:
            args.model_names = [f"model_{i}" for i in range(len(args.model_paths))]
        
        # Generate algorithms if not provided
        if not args.algorithms:
            args.algorithms = ["sac"] * len(args.model_paths)
        
        # Load data
        df = load_data(args.data_path)
        
        # Load models and create strategies
        strategies = {}
        
        for i, (model_path, model_name, algorithm) in enumerate(zip(args.model_paths, args.model_names, args.algorithms)):
            # Load model
            model = load_model(
                model_path=model_path,
                algorithm=algorithm,
                device=args.device
            )
            
            # Create strategy
            strategy = create_strategy(
                model=model,
                algorithm=algorithm,
                feature_cols=args.feature_cols,
                price_col=args.price_col,
                date_col=args.date_col,
                symbol_col=args.symbol_col,
                device=args.device
            )
            
            # Add to strategies
            strategies[model_name] = strategy
        
        # Compare strategies
        results = compare_strategies(
            strategies=strategies,
            test_df=df,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            risk_free_rate=args.risk_free_rate
        )
        
        # Print comparison
        print("\nComparison Results:")
        print(results["comparison"])
        
        # Save results
        if args.output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            
            # Save results
            with open(args.output_file, "w") as f:
                # Convert DataFrame to dict
                comparison_dict = results["comparison"].to_dict()
                
                # Convert numpy values to Python types
                for metric, values in comparison_dict.items():
                    for model, value in values.items():
                        if isinstance(value, np.ndarray):
                            comparison_dict[metric][model] = value.tolist()
                        elif isinstance(value, np.float64):
                            comparison_dict[metric][model] = float(value)
                        elif isinstance(value, np.int64):
                            comparison_dict[metric][model] = int(value)
                
                json.dump(comparison_dict, f, indent=4)
            
            print(f"\nResults saved to: {args.output_file}")
        
        # Plot comparison
        if args.plot:
            try:
                plot_comparison(results)
            except Exception as e:
                logger.error(f"Error plotting comparison: {e}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
