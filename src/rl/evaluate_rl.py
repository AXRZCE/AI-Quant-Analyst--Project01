"""
Evaluate a trained reinforcement learning agent for trading.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ray, but don't fail if it's not available
try:
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.agents.sac import SACTrainer
    from ray.tune.registry import register_env
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray is not available. Please install it with 'pip install ray[rllib]'.")
    RAY_AVAILABLE = False

# Import local modules
from src.rl.env import TradingEnv
from src.rl.train_rl import env_creator

def load_agent(
    checkpoint_path: str,
    algorithm: str = "ppo",
    env_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Load a trained agent from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        algorithm: RL algorithm used for training
        env_config: Environment configuration
        
    Returns:
        Trained agent
    """
    if not RAY_AVAILABLE:
        logger.error("Ray is not available. Please install it with 'pip install ray[rllib]'.")
        return None
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("trading_env", lambda cfg: env_creator(cfg))
    
    # Create default environment config if not provided
    if env_config is None:
        env_config = {}
    
    # Create agent config
    config = {
        "env": "trading_env",
        "env_config": env_config,
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 0
    }
    
    # Create agent
    if algorithm.lower() == "ppo":
        agent = PPOTrainer(config=config)
    elif algorithm.lower() == "sac":
        agent = SACTrainer(config=config)
    else:
        logger.error(f"Unknown algorithm: {algorithm}")
        return None
    
    # Load checkpoint
    agent.restore(checkpoint_path)
    logger.info(f"Agent loaded from {checkpoint_path}")
    
    return agent

def evaluate_agent(
    agent: Any,
    env: TradingEnv,
    num_episodes: int = 1,
    render: bool = False
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Evaluate a trained agent on an environment.
    
    Args:
        agent: Trained agent
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        List of episode rewards and list of episode info
    """
    episode_rewards = []
    episode_infos = []
    
    for i in range(num_episodes):
        logger.info(f"Evaluating episode {i+1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Run episode
        while not done:
            # Get action from agent
            action = agent.compute_action(obs)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            # Update episode reward
            episode_reward += reward
            step += 1
            
            # Render environment
            if render:
                env.render()
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_infos.append({
            "episode_reward": episode_reward,
            "episode_length": step,
            "final_portfolio_value": info["portfolio_value"]
        })
        
        logger.info(f"Episode {i+1} reward: {episode_reward:.2f}, length: {step}, final portfolio value: ${info['portfolio_value']:.2f}")
    
    # Calculate average reward
    avg_reward = np.mean(episode_rewards)
    logger.info(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return episode_rewards, episode_infos

def generate_predictions(
    agent: Any,
    env: TradingEnv,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate predictions using a trained agent.
    
    Args:
        agent: Trained agent
        env: Environment to generate predictions for
        output_path: Path to save predictions to
        
    Returns:
        DataFrame with predictions
    """
    # Reset environment
    obs = env.reset()
    done = False
    
    # Initialize lists to store data
    steps = []
    actions = []
    rewards = []
    portfolio_values = []
    positions = []
    timestamps = []
    symbols = []
    
    # Run episode
    while not done:
        # Get action from agent
        action = agent.compute_action(obs)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        
        # Record step data
        steps.append(env.current_step)
        actions.append(action)
        rewards.append(reward)
        portfolio_values.append(info["portfolio_value"])
        
        # Record positions for each symbol
        for symbol in env.symbols:
            positions.append(env.positions[symbol])
            symbols.append(symbol)
            
            # Get timestamp for this step
            symbol_df = env.df[env.df[env.symbol_col] == symbol]
            if env.current_step - 1 < len(symbol_df):
                timestamp = symbol_df.iloc[env.current_step - 1][env.date_col]
            else:
                timestamp = symbol_df.iloc[-1][env.date_col]
            timestamps.append(timestamp)
    
    # Get portfolio history
    portfolio_df = env.get_portfolio_history()
    
    # Create predictions DataFrame
    predictions = []
    for symbol in env.symbols:
        symbol_df = env.df[env.df[env.symbol_col] == symbol].copy()
        
        # Add predictions
        symbol_actions = portfolio_df[f"action_{symbol}"].values
        symbol_df["pred_rl"] = symbol_actions
        
        # Add to list
        predictions.append(symbol_df)
    
    # Combine predictions
    predictions_df = pd.concat(predictions, ignore_index=True)
    
    # Save predictions if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_parquet(output_path)
        logger.info(f"Predictions saved to {output_path}")
    
    return predictions_df

def evaluate(
    checkpoint_path: str,
    data_path: str,
    feature_cols: List[str],
    algorithm: str = "ppo",
    output_path: Optional[str] = None,
    num_episodes: int = 1,
    render: bool = False,
    **env_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a trained agent on a dataset.
    
    Args:
        checkpoint_path: Path to the checkpoint
        data_path: Path to the data file
        feature_cols: Feature column names
        algorithm: RL algorithm used for training
        output_path: Path to save predictions to
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        env_kwargs: Additional environment arguments
        
    Returns:
        Evaluation results
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return {"status": "error", "message": "Checkpoint not found"}
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return {"status": "error", "message": "Data file not found"}
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data from {data_path} with {len(df)} rows")
    
    # Create environment config
    env_config = {
        "data_path": data_path,
        "feature_cols": feature_cols,
        **env_kwargs
    }
    
    # Load agent
    agent = load_agent(checkpoint_path, algorithm, env_config)
    if agent is None:
        return {"status": "error", "message": "Failed to load agent"}
    
    # Create environment
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        **env_kwargs
    )
    
    # Evaluate agent
    episode_rewards, episode_infos = evaluate_agent(agent, env, num_episodes, render)
    
    # Generate predictions
    if output_path:
        predictions_df = generate_predictions(agent, env, output_path)
    
    # Return results
    return {
        "status": "success",
        "episode_rewards": episode_rewards,
        "episode_infos": episode_infos,
        "average_reward": np.mean(episode_rewards),
        "average_portfolio_value": np.mean([info["final_portfolio_value"] for info in episode_infos])
    }

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Evaluate agent
    result = evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        feature_cols=args.feature_cols,
        algorithm=args.algorithm,
        output_path=args.output_path,
        num_episodes=args.num_episodes,
        render=args.render,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        reward_scaling=args.reward_scaling,
        window_size=args.window_size,
        symbol_col=args.symbol_col,
        price_col=args.price_col,
        date_col=args.date_col
    )
    
    # Check result
    if result["status"] == "error":
        logger.error(f"Evaluation failed: {result['message']}")
        return
    
    # Print results
    logger.info(f"Evaluation completed successfully.")
    logger.info(f"Average reward: {result['average_reward']:.2f}")
    logger.info(f"Average portfolio value: ${result['average_portfolio_value']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained reinforcement learning agent for trading.")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm used for training")
    
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
    
    # Evaluation arguments
    parser.add_argument("--output-path", type=str, default=None, help="Path to save predictions to")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment")
    
    args = parser.parse_args()
    main(args)
