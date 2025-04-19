"""
Soft Actor-Critic (SAC) implementation for trading.

This module provides an implementation of the Soft Actor-Critic algorithm
for reinforcement learning in trading environments.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Replay buffer for SAC."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum capacity of the buffer
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])

        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    """Q-Network for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        init_w: float = 3e-3
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            init_w: Weight initialization range
        """
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear6.weight.data.uniform_(-init_w, init_w)
        self.linear6.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (Q1, Q2)
        """
        x = torch.cat([state, action], 1)

        # Q1
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        # Q2
        q2 = F.relu(self.linear4(x))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2


class GaussianPolicy(nn.Module):
    """Gaussian policy for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        init_w: float = 3e-3,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize Gaussian policy.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            init_w: Weight initialization range
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(GaussianPolicy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            Tuple of (mean, log_std)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t)

        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class SAC:
    """Soft Actor-Critic algorithm."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SAC.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            lr: Learning rate
            automatic_entropy_tuning: Whether to automatically tune entropy
            device: Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device

        # Initialize critic
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize actor
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Training info
        self.total_steps = 0
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action from policy.

        Args:
            state: Current state
            evaluate: Whether to evaluate or explore

        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update parameters using batch from replay buffer.

        Args:
            batch_size: Batch size for update

        Returns:
            Dictionary with loss information
        """
        batch_size = batch_size or self.batch_size

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_new, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = torch.tensor(0.0).to(self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
        else:
            # Keep alpha as the initialized value
            pass

        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # Record losses
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        self.alpha_losses.append(alpha_loss.item())

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha
        }

    def train(
        self,
        env,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        eval_interval: int = 10,
        num_eval_episodes: int = 5,
        warmup_steps: int = 1000,
        updates_per_step: int = 1,
        save_path: Optional[str] = None,
        save_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the agent.

        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            eval_interval: Interval between evaluations
            num_eval_episodes: Number of episodes for evaluation
            warmup_steps: Number of steps to collect experience before training
            updates_per_step: Number of updates per step
            save_path: Path to save model
            save_interval: Interval between model saves
            verbose: Whether to print progress

        Returns:
            Dictionary with training information
        """
        # Training metrics
        episode_rewards = []
        eval_rewards = []

        # Initial exploration
        if warmup_steps > 0 and verbose:
            logger.info(f"Collecting {warmup_steps} warmup steps...")

        state, _ = env.reset()
        for step in range(warmup_steps):
            action = np.random.uniform(-1, 1, size=self.action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            if done:
                state, _ = env.reset()

        # Main training loop
        if verbose:
            logger.info("Starting training...")

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update parameters
                if self.replay_buffer.size >= self.batch_size:
                    for _ in range(updates_per_step):
                        self.update_parameters()

                state = next_state
                episode_reward += reward
                self.total_steps += 1

                if done:
                    break

            # Record episode reward
            episode_rewards.append(episode_reward)
            self.episode_rewards.append(episode_reward)

            # Evaluate agent
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(env, num_eval_episodes)
                eval_rewards.append(eval_reward)

                if verbose:
                    logger.info(f"Episode {episode+1}/{num_episodes} | " +
                                f"Episode Reward: {episode_reward:.2f} | " +
                                f"Eval Reward: {eval_reward:.2f}")
            elif verbose:
                logger.info(f"Episode {episode+1}/{num_episodes} | " +
                            f"Episode Reward: {episode_reward:.2f}")

            # Save model
            if save_path and (episode + 1) % save_interval == 0:
                self.save(os.path.join(save_path, f"sac_episode_{episode+1}.pt"))

        # Final save
        if save_path:
            self.save(os.path.join(save_path, "sac_final.pt"))

        return {
            "episode_rewards": episode_rewards,
            "eval_rewards": eval_rewards,
            "critic_losses": self.critic_losses,
            "actor_losses": self.actor_losses,
            "alpha_losses": self.alpha_losses
        }

    def evaluate(self, env, num_episodes: int = 10) -> float:
        """
        Evaluate the agent.

        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate for

        Returns:
            Average reward over episodes
        """
        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)

        return np.mean(rewards)

    def save(self, path: str) -> None:
        """
        Save model.

        Args:
            path: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.automatic_entropy_tuning else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
            "critic_losses": self.critic_losses,
            "actor_losses": self.actor_losses,
            "alpha_losses": self.alpha_losses,
            "hyperparameters": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "buffer_capacity": self.buffer_capacity,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "lr": self.lr,
                "automatic_entropy_tuning": self.automatic_entropy_tuning
            }
        }, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model parameters
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor.load_state_dict(checkpoint["actor"])

        # Load optimizer parameters
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        # Load alpha parameters
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            self.alpha = self.log_alpha.exp().item()
        else:
            # If automatic_entropy_tuning is False, use the saved alpha value
            if "alpha" in checkpoint["hyperparameters"]:
                self.alpha = checkpoint["hyperparameters"]["alpha"]

        # Load training info
        self.total_steps = checkpoint["total_steps"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.critic_losses = checkpoint["critic_losses"]
        self.actor_losses = checkpoint["actor_losses"]
        self.alpha_losses = checkpoint["alpha_losses"]

        logger.info(f"Model loaded from {path}")


def train_sac(
    env,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    buffer_capacity: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    lr: float = 3e-4,
    automatic_entropy_tuning: bool = True,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    eval_interval: int = 10,
    num_eval_episodes: int = 5,
    warmup_steps: int = 1000,
    updates_per_step: int = 1,
    save_path: Optional[str] = None,
    save_interval: int = 100,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[SAC, Dict[str, List[float]]]:
    """
    Train SAC agent.

    Args:
        env: Environment to train on
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Dimension of hidden layers
        buffer_capacity: Capacity of replay buffer
        batch_size: Batch size for training
        gamma: Discount factor
        tau: Target network update rate
        alpha: Temperature parameter for entropy
        lr: Learning rate
        automatic_entropy_tuning: Whether to automatically tune entropy
        num_episodes: Number of episodes to train for
        max_steps_per_episode: Maximum steps per episode
        eval_interval: Interval between evaluations
        num_eval_episodes: Number of episodes for evaluation
        warmup_steps: Number of steps to collect experience before training
        updates_per_step: Number of updates per step
        save_path: Path to save model
        save_interval: Interval between model saves
        verbose: Whether to print progress
        device: Device to use for training

    Returns:
        Tuple of (agent, training_info)
    """
    # Create agent
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

    # Train agent
    training_info = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes,
        warmup_steps=warmup_steps,
        updates_per_step=updates_per_step,
        save_path=save_path,
        save_interval=save_interval,
        verbose=verbose
    )

    return agent, training_info
