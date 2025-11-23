"""
PPO (Proximal Policy Optimization) Agent for Drone Control

This module implements a PPO agent optimized for continuous control tasks
like drone flight. The implementation includes:
- Actor-Critic neural network architecture
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function clipping
- Entropy bonus for exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "tanh"

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_clip: float = 0.2  # Value function clipping
    entropy_coef: float = 0.02  # Entropy bonus coefficient (increased for exploration)
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 1.0  # Gradient clipping (increased for stability)

    # Training parameters
    n_steps: int = 2048  # Steps per environment per update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # PPO epochs per update
    normalize_advantages: bool = True

    # Action space
    action_std_init: float = 0.5  # Initial action standard deviation
    action_std_min: float = 0.1  # Minimum action standard deviation
    action_std_decay: float = 0.95  # Std decay per update (faster convergence)


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.

    The actor outputs mean actions, with a learnable standard deviation.
    The critic estimates state values.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        action_std_init: float = 0.5
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Select activation function
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.Tanh

        # Build shared feature extractor
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers) if layers else nn.Identity()

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            act_fn(),
            nn.Linear(hidden_sizes[-1], action_dim),
            nn.Tanh()  # Output in [-1, 1] for reversible motor commands
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            act_fn(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * np.log(action_std_init)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Smaller initialization for output layers
        for module in [self.actor[-2], self.critic[-1]]:
            nn.init.orthogonal_(module.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        features = self.shared(obs)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Estimated state value
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Clip action to valid range (reversible motors use [-1, 1])
        action = torch.clamp(action, -1, 1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_prob: Log probabilities of actions
            value: State values
            entropy: Policy entropy
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()

        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        # Pre-allocate tensors
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        # Don't add if buffer is full - must call reset() or update first
        if self.ptr >= self.buffer_size:
            self.full = True
            return

        self.observations[self.ptr] = torch.from_numpy(obs).to(self.device)
        self.actions[self.ptr] = torch.from_numpy(action).to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float
    ):
        """Compute returns and GAE advantages."""
        last_gae = 0
        n = self.ptr if not self.full else self.buffer_size

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = (
                self.rewards[t].item() +
                gamma * next_value * next_non_terminal -
                self.values[t].item()
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Generate random minibatches."""
        n = self.ptr if not self.full else self.buffer_size
        indices = torch.randperm(n, device=self.device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'values': self.values[batch_indices]
            }

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.full = False


class PPOAgent:
    """
    PPO Agent for drone control.

    This agent implements the PPO algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation
    - Value function clipping
    - Entropy regularization
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = "auto"
    ):
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize actor-critic network
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            action_std_init=self.config.action_std_init
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )

        # Training statistics
        self.total_steps = 0
        self.updates = 0

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)

        return (
            action.cpu().numpy().squeeze(0),
            {'log_prob': log_prob.item(), 'value': value.item()}
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, reward, value, log_prob, done)
        self.total_steps += 1

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """Perform PPO update."""
        # Compute last value for GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(obs_tensor)
            last_value = last_value.item()

        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda
        )

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = self.buffer.advantages[:self.buffer.ptr]
            self.buffer.advantages[:self.buffer.ptr] = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            )

        # PPO update epochs
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch['observations'],
                    batch['actions']
                )

                # Compute ratio
                ratio = torch.exp(log_probs - batch['log_probs'])

                # Clipped surrogate objective
                advantages = batch['advantages']
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                values_clipped = batch['values'] + torch.clamp(
                    values - batch['values'],
                    -self.config.value_clip,
                    self.config.value_clip
                )
                value_loss1 = (values - batch['returns']) ** 2
                value_loss2 = (values_clipped - batch['returns']) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        # Reset buffer
        self.buffer.reset()

        # Decay action std
        with torch.no_grad():
            self.policy.log_std.data = torch.clamp(
                self.policy.log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min)
            )

        self.updates += 1

        return {
            'loss': total_loss / (self.config.n_epochs * (self.config.n_steps // self.config.batch_size)),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'action_std': self.policy.log_std.exp().mean().item()
        }

    def save(self, path: str):
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'total_steps': self.total_steps,
            'updates': self.updates
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> 'PPOAgent':
        """Create agent from checkpoint file."""
        checkpoint = torch.load(path, map_location='cpu')

        config = PPOConfig(**checkpoint['config'])
        agent = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            config=config,
            device=device
        )
        agent.load(path)

        return agent
