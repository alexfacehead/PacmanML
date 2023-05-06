import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models.ghost_model import GhostModel
from utils.replay_buffer import ReplayBuffer
from typing import Tuple

class GhostAgent:
    # Add num_stacked_frames parameter to the constructor
    def __init__(self, state_size: Tuple[int, int], action_size: int, hidden_size: int = 128, batch_size: int = 64, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01, buffer_size: int = 10000, device: str = "cpu", num_stacked_frames: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.device = device
        self.stacked_frames = deque(maxlen=num_stacked_frames)
        self.num_stacked_frames = num_stacked_frames

        self.model = GhostModel(state_size, action_size, hidden_size).to(device)
        self.target_model = GhostModel(state_size, action_size, hidden_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state: np.ndarray) -> int:
        state = self.preprocess_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        img = Image.fromarray(state)
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((84, 84))  # Resize to 84x84
        processed_state = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
        return processed_state