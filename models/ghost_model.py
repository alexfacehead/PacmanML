import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostDQN(nn.Module):
    """DQN for Ghost team agent.

    Input:  (batch, in_channels, 31, 28) — multi-channel grid state
    Output: (batch, num_ghosts, num_actions) — Q-values for each ghost's actions

    Uses shared convolutional layers with separate FC heads per ghost.

    Architecture:
        conv1: Conv2d(in_channels, 32, 3, padding=1) + ReLU  -> (32, 31, 28)
        conv2: Conv2d(32, 64, 3, padding=1) + ReLU           -> (64, 31, 28)
        conv3: Conv2d(64, 64, 3, stride=2) + ReLU            -> (64, 15, 13)
        flatten: 64 * 15 * 13 = 12480
        shared_fc: Linear(12480, 256) + ReLU
        ghost_heads: num_ghosts x Linear(256, num_actions)
    """

    def __init__(self, in_channels: int = 6, num_ghosts: int = 4,
                 num_actions: int = 5):
        super().__init__()
        self.num_ghosts = num_ghosts

        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        # Shared fully-connected layer
        self.shared_fc = nn.Linear(64 * 15 * 13, 256)

        # Separate head for each ghost
        self.ghost_heads = nn.ModuleList([
            nn.Linear(256, num_actions) for _ in range(num_ghosts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.shared_fc(x))

        # Each head produces (batch, num_actions); stack to (batch, num_ghosts, num_actions)
        ghost_qs = torch.stack([head(x) for head in self.ghost_heads], dim=1)
        return ghost_qs
