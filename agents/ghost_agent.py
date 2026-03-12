"""DQN agent controlling all 4 ghosts as a team."""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.ghost_model import GhostDQN


def _auto_device() -> torch.device:
    """Select best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GhostReplayBuffer:
    """Replay buffer for the ghost team.

    Stores transitions where actions is a list of 4 ints (one per ghost).
    """

    def __init__(self, capacity: int):
        from collections import deque
        self.buffer: deque = deque(maxlen=capacity)
        self._batch_states: np.ndarray | None = None
        self._batch_next: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        actions: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = random.sample(range(len(self.buffer)), batch_size)

        sample0 = self.buffer[indices[0]]
        shape = sample0[0].shape
        if self._batch_states is None or self._batch_states.shape[0] != batch_size:
            self._batch_states = np.empty((batch_size, *shape), dtype=np.float32)
            self._batch_next = np.empty((batch_size, *shape), dtype=np.float32)

        num_ghosts = len(sample0[1])
        actions = np.empty((batch_size, num_ghosts), dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)

        for i, idx in enumerate(indices):
            s, a, r, ns, d = self.buffer[idx]
            self._batch_states[i] = s
            actions[i] = a
            rewards[i] = r
            self._batch_next[i] = ns
            dones[i] = float(d)

        return self._batch_states, actions, rewards, self._batch_next, dones

    def __len__(self) -> int:
        return len(self.buffer)


class GhostAgent:
    """DQN agent controlling all 4 ghosts as a team.

    The network outputs Q-values for each ghost independently (shared backbone).
    Uses Double-DQN with epsilon-greedy exploration per ghost.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_ghosts: int = 4,
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

        self.num_ghosts = num_ghosts
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.model = GhostDQN(in_channels, num_ghosts, num_actions).to(self.device)
        self.target_model = GhostDQN(in_channels, num_ghosts, num_actions).to(
            self.device
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer (stores multi-ghost actions)
        self.buffer = GhostReplayBuffer(buffer_size)

        # Step counter
        self.steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> List[int]:
        """Epsilon-greedy action selection for each ghost.

        Args:
            state: Array of shape (C, H, W).

        Returns:
            List of 4 action indices.
        """
        if random.random() < self.epsilon:
            # All ghosts random
            return [random.randint(0, self.num_actions - 1) for _ in range(self.num_ghosts)]

        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.model(t)  # (1, num_ghosts, num_actions)
            return q_values[0].argmax(dim=1).tolist()

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        actions: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.add(state, actions, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """Sample a batch and perform one gradient step (Double DQN).

        Returns:
            Loss value, or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)          # (B, 4)
        rewards_t = torch.from_numpy(rewards).to(self.device)          # (B,)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values: (B, num_ghosts, num_actions)
        q_all = self.model(states_t)
        # Gather selected actions for each ghost: (B, num_ghosts)
        q_selected = q_all.gather(2, actions_t.unsqueeze(2)).squeeze(2)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.model(next_states_t)  # (B, G, A)
            best_actions = next_q_online.argmax(dim=2, keepdim=True)  # (B, G, 1)
            next_q_target = self.target_model(next_states_t)
            next_q = next_q_target.gather(2, best_actions).squeeze(2)  # (B, G)
            # Mean across ghosts for the team target
            next_q_mean = next_q.mean(dim=1)  # (B,)
            target = rewards_t + self.gamma * next_q_mean * (1.0 - dones_t)

        # Loss: average across ghosts
        q_mean = q_selected.mean(dim=1)  # (B,)
        loss = self.loss_fn(q_mean, target)

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
