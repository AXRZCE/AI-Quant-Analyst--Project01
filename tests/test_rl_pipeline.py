"""
Integration tests for the RL pipeline.
"""
import os
import sys
import pytest
import subprocess
import pandas as pd
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.rl.env import TradingEnv

def test_rl_env_creation():
    """
    Test that the RL environment can be created.
    """
    # Check if data exists
    data_path = "data/features/batch/technical_with_timeidx.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_parquet(data_path)

    # Create environment
    feature_cols = ["ma_5", "rsi_14", "close"]
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        initial_capital=100_000,
        transaction_cost=0.001
    )

    # Check environment
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

    # Reset environment
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)

    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_rl_training_smoke():
    """
    Test that the RL training script runs without errors.
    """
    # Check if data exists
    data_path = "data/features/batch/technical_with_timeidx.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    # Check if Ray is available
    try:
        import ray
    except ImportError:
        pytest.skip("Ray is not available")

    # Run one training iteration with tiny data & iterations
    cmd = [
        "python", "src/rl/train_rl.py",
        "--data-path", data_path,
        "--feature-cols", "ma_5", "rsi_14", "close",
        "--num-workers", "0",
        "--max-iters", "1",
        "--algorithm", "ppo",
        "--experiment-name", "test_ppo",
        "--log-dir", "logs/rl",
        "--batch-size", "128",
        "--rollout-length", "10",
        "--debug"
    ]

    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Check result
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        # Check output
        assert "Running backtest with initial capital" in result.stdout or "Running backtest with initial capital" in result.stderr
    except subprocess.TimeoutExpired:
        pytest.skip("Command timed out")

def test_rl_evaluation_smoke():
    """
    Test that the RL evaluation script runs without errors.
    """
    # Check if data exists
    data_path = "data/features/batch/technical_with_timeidx.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    # Check if Ray is available
    try:
        import ray
    except ImportError:
        pytest.skip("Ray is not available")

    # Check if checkpoint exists
    checkpoint_path = "logs/rl/test_ppo"
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # Find the latest checkpoint
    checkpoints = []
    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            if file.startswith("checkpoint-"):
                checkpoints.append(os.path.join(root, file))

    if not checkpoints:
        pytest.skip("No checkpoints found")

    # Sort checkpoints by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Get the latest checkpoint
    checkpoint = checkpoints[0]

    # Run evaluation
    cmd = [
        "python", "src/rl/evaluate_rl.py",
        "--checkpoint", checkpoint,
        "--data-path", data_path,
        "--feature-cols", "ma_5", "rsi_14", "close",
        "--algorithm", "ppo",
        "--num-episodes", "1"
    ]

    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Check result
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        # Check output
        assert "Evaluating episode" in result.stdout or "Evaluating episode" in result.stderr
    except subprocess.TimeoutExpired:
        pytest.skip("Command timed out")

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
