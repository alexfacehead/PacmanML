"""DQN agent controlling Pacman."""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.pacman_model import PacmanDQN
from utils.replay_buffer import ReplayBuffer


def _auto_device() -> torch.device:
    """Select best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PacmanAgent:
    """DQN agent controlling Pacman.

    Uses Double-DQN with a target network and epsilon-greedy exploration.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_actions: int = 5,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "auto",
    ):
        # Device
        if device == "auto":
            self.device = _auto_device()
        else:
            self.device = torch.device(device)

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.model = PacmanDQN(in_channels, num_actions).to(self.device)
        self.target_model = PacmanDQN(in_channels, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Step counter for target updates
        self.steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: Array of shape (C, H, W).

        Returns:
            Action index in [0, num_actions).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.model(t)
            return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """Sample a batch and perform one gradient step (Double DQN).

        Returns:
            Loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values
        q_values = self.model(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_q_online = self.model(next_states_t)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_model(next_states_t)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Defer .item() to avoid MPS→CPU sync; store on tensor
        self._last_loss = loss.detach()
        return self._last_loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state from *path*."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
