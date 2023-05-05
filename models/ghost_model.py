import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GhostModel(nn.Module):
    def __init__(self, state_size: Tuple[int, int], action_size: int, hidden_size: int = 128):
        super(GhostModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * state_size[1] * state_size[2], hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
