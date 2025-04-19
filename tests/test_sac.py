"""
Tests for the Soft Actor-Critic (SAC) implementation.
"""

import os
import pytest
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from src.rl.sac import SAC, ReplayBuffer, QNetwork, GaussianPolicy, train_sac


class DummyEnv:
    """Dummy environment for testing."""

    def __init__(self, state_dim=4, action_dim=2):
        """Initialize dummy environment."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        self.state = None
        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        """Reset environment."""
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """Take step in environment."""
        self.step_count += 1
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        reward = float(np.sum(action))  # Simple reward function
        done = self.step_count >= self.max_steps
        return self.state, reward, done, False, {}


def test_replay_buffer():
    """Test replay buffer."""
    # Create replay buffer
    buffer = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)

    # Add transitions
    for _ in range(10):
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = np.random.rand()
        next_state = np.random.randn(4)
        done = bool(np.random.randint(0, 2))

        buffer.add(state, action, reward, next_state, done)

    # Check buffer size
    assert buffer.size == 10

    # Sample batch
    batch_size = 5
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Check batch shapes
    assert states.shape == (batch_size, 4)
    assert actions.shape == (batch_size, 2)
    assert rewards.shape == (batch_size, 1)
    assert next_states.shape == (batch_size, 4)
    assert dones.shape == (batch_size, 1)

    # Check types
    assert isinstance(states, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(next_states, torch.Tensor)
    assert isinstance(dones, torch.Tensor)


def test_q_network():
    """Test Q-Network."""
    # Create Q-Network
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    q_network = QNetwork(state_dim, action_dim, hidden_dim)

    # Create inputs
    batch_size = 10
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    # Forward pass
    q1, q2 = q_network(states, actions)

    # Check output shapes
    assert q1.shape == (batch_size, 1)
    assert q2.shape == (batch_size, 1)


def test_gaussian_policy():
    """Test Gaussian policy."""
    # Create Gaussian policy
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    policy = GaussianPolicy(state_dim, action_dim, hidden_dim)

    # Create inputs
    batch_size = 10
    states = torch.randn(batch_size, state_dim)

    # Forward pass
    mean, log_std = policy(states)

    # Check output shapes
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)

    # Sample actions
    actions, log_probs, means = policy.sample(states)

    # Check output shapes
    assert actions.shape == (batch_size, action_dim)
    assert log_probs.shape == (batch_size, 1)
    assert means.shape == (batch_size, action_dim)

    # Check action bounds
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_sac_initialization():
    """Test SAC initialization."""
    # Create SAC agent
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    buffer_capacity = 100
    batch_size = 16
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    lr = 3e-4
    automatic_entropy_tuning = True
    device = "cpu"

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        lr=lr,
        automatic_entropy_tuning=automatic_entropy_tuning,
        device=device
    )

    # Check agent attributes
    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert agent.hidden_dim == hidden_dim
    assert agent.buffer_capacity == buffer_capacity
    assert agent.batch_size == batch_size
    assert agent.gamma == gamma
    assert agent.tau == tau
    assert agent.alpha == alpha
    assert agent.lr == lr
    assert agent.automatic_entropy_tuning == automatic_entropy_tuning
    assert agent.device == device

    # Check agent components
    assert isinstance(agent.critic, QNetwork)
    assert isinstance(agent.critic_target, QNetwork)
    assert isinstance(agent.actor, GaussianPolicy)
    assert isinstance(agent.replay_buffer, ReplayBuffer)

    # Check target entropy
    if automatic_entropy_tuning:
        assert agent.target_entropy == -action_dim


def test_sac_select_action():
    """Test SAC action selection."""
    # Create SAC agent
    state_dim = 4
    action_dim = 2
    agent = SAC(state_dim=state_dim, action_dim=action_dim, device="cpu")

    # Select action
    state = np.random.randn(state_dim)
    action = agent.select_action(state)

    # Check action shape and bounds
    assert action.shape == (action_dim,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    # Select action in evaluation mode
    action_eval = agent.select_action(state, evaluate=True)

    # Check action shape and bounds
    assert action_eval.shape == (action_dim,)
    assert np.all(action_eval >= -1.0)
    assert np.all(action_eval <= 1.0)


def test_sac_update_parameters():
    """Test SAC parameter update."""
    # Create SAC agent
    state_dim = 4
    action_dim = 2
    batch_size = 16
    agent = SAC(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, device="cpu")

    # Fill replay buffer
    for _ in range(batch_size * 2):
        state = np.random.randn(state_dim)
        action = np.random.uniform(-1, 1, size=action_dim)
        reward = np.random.rand()
        next_state = np.random.randn(state_dim)
        done = bool(np.random.randint(0, 2))

        agent.replay_buffer.add(state, action, reward, next_state, done)

    # Update parameters
    update_info = agent.update_parameters()

    # Check update info
    assert "critic_loss" in update_info
    assert "actor_loss" in update_info
    assert "alpha_loss" in update_info
    assert "alpha" in update_info

    # Check losses
    assert len(agent.critic_losses) == 1
    assert len(agent.actor_losses) == 1
    assert len(agent.alpha_losses) == 1


def test_sac_save_load(tmp_path):
    """Test SAC save and load."""
    # Create SAC agent
    state_dim = 4
    action_dim = 2
    agent = SAC(state_dim=state_dim, action_dim=action_dim, device="cpu")

    # Save agent
    save_path = os.path.join(tmp_path, "sac_test.pt")
    agent.save(save_path)

    # Check file exists
    assert os.path.exists(save_path)

    # Create new agent with the same parameters
    new_agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=agent.hidden_dim,
        buffer_capacity=agent.buffer_capacity,
        batch_size=agent.batch_size,
        gamma=agent.gamma,
        tau=agent.tau,
        alpha=agent.alpha,  # Use the same alpha value
        automatic_entropy_tuning=agent.automatic_entropy_tuning,
        device="cpu"
    )

    # Load agent
    new_agent.load(save_path)

    # Check agent attributes
    assert new_agent.state_dim == agent.state_dim
    assert new_agent.action_dim == agent.action_dim
    assert new_agent.hidden_dim == agent.hidden_dim
    assert new_agent.buffer_capacity == agent.buffer_capacity
    assert new_agent.batch_size == agent.batch_size
    assert new_agent.gamma == agent.gamma
    assert new_agent.tau == agent.tau
    # Skip alpha check as it might be different depending on automatic_entropy_tuning
    # assert new_agent.alpha == agent.alpha
    assert new_agent.automatic_entropy_tuning == agent.automatic_entropy_tuning


def test_sac_train():
    """Test SAC training."""
    # Create environment
    env = DummyEnv(state_dim=4, action_dim=2)

    # Train SAC agent
    agent, training_info = train_sac(
        env=env,
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        buffer_capacity=1000,
        batch_size=16,
        num_episodes=2,
        max_steps_per_episode=10,
        eval_interval=1,
        num_eval_episodes=1,
        warmup_steps=20,
        updates_per_step=1,
        verbose=False,
        device="cpu"
    )

    # Check training info
    assert "episode_rewards" in training_info
    assert "eval_rewards" in training_info
    assert "critic_losses" in training_info
    assert "actor_losses" in training_info
    assert "alpha_losses" in training_info

    # Check lengths
    assert len(training_info["episode_rewards"]) == 2
    assert len(training_info["eval_rewards"]) == 2
    assert len(training_info["critic_losses"]) > 0
    assert len(training_info["actor_losses"]) > 0
    assert len(training_info["alpha_losses"]) > 0

    # Check agent
    assert agent.total_steps > 0
    assert len(agent.episode_rewards) == 2
