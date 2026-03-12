import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Standard experience replay buffer for DQN training.

    Stores transitions as (state, action, reward, next_state, done) tuples
    where states are numpy arrays of shape (6, 31, 28).
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)
        self._batch_states: np.ndarray | None = None
        self._batch_next: np.ndarray | None = None

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Uses pre-allocated arrays to avoid repeated allocation.
        """
        indices = random.sample(range(len(self.buffer)), batch_size)

        # Pre-allocate or reuse batch arrays for states (largest allocation)
        sample0 = self.buffer[indices[0]]
        shape = sample0[0].shape
        if self._batch_states is None or self._batch_states.shape[0] != batch_size:
            self._batch_states = np.empty((batch_size, *shape), dtype=np.float32)
            self._batch_next = np.empty((batch_size, *shape), dtype=np.float32)

        actions = np.empty(batch_size, dtype=np.int64)
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
