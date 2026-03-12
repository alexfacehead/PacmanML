"""Gymnasium environment wrapper for the PacmanML game engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from pacman.core.game import Game


# Project root is one level above this file's parent directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LEVEL = str(_PROJECT_ROOT / "pacman" / "levels" / "level_1.txt")


class PacmanEnv(gymnasium.Env):
    """Gymnasium environment for adversarial Pacman ML training.

    Supports two perspectives:
    - agent_role="pacman": agent controls Pacman, ghosts use built-in AI
    - agent_role="ghosts": agent controls all 4 ghosts, Pacman uses built-in AI
      or a provided policy

    The action space for "pacman" is Discrete(5) -- UP/DOWN/LEFT/RIGHT/STAY
    The action space for "ghosts" is MultiDiscrete([5,5,5,5]) -- one action per
    ghost
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        level_file: Optional[str] = None,
        agent_role: str = "pacman",
        opponent_policy: Optional[Callable[[np.ndarray], Any]] = None,
        max_steps: int = 3000,
        frame_skip: int = 1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if agent_role not in ("pacman", "ghosts"):
            raise ValueError(f"agent_role must be 'pacman' or 'ghosts', got '{agent_role}'")

        self.level_file = level_file or _DEFAULT_LEVEL
        self.agent_role = agent_role
        self.opponent_policy = opponent_policy
        self.max_steps = max_steps
        self.frame_skip = max(1, frame_skip)
        self.render_mode = render_mode

        # Create the game engine (headless for training).
        self._game = Game(headless=True)
        self._game.load_level(self.level_file)

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, 31, 28), dtype=np.float32
        )

        if self.agent_role == "pacman":
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])

        # Step counter
        self._steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._game.reset()
        self._steps = 0

        obs = self._game.get_state()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0

        for _ in range(self.frame_skip):
            pacman_action, ghost_actions = self._resolve_actions(action)
            self._game.step(pacman_action=pacman_action, ghost_actions=ghost_actions)
            self._steps += 1

            # Accumulate reward across skipped frames.
            if self.agent_role == "pacman":
                total_reward += self._game.get_reward_pacman()
            else:
                total_reward += self._game.get_reward_ghost()

            if self._game.is_done() or self._steps >= self.max_steps:
                break

        obs = self._game.get_state()
        terminated = self._game.game_over or self._game.level_complete
        truncated = (not terminated) and (self._steps >= self.max_steps)
        info = self._build_info()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render is a no-op in headless mode.

        Returns the state grid as an array when render_mode is 'rgb_array'.
        """
        if self.render_mode == "rgb_array":
            return self._game.get_state()
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_actions(
        self, action: int | np.ndarray
    ) -> tuple[Optional[int], Optional[list[Optional[int]]]]:
        """Convert the agent action into game engine arguments.

        Depending on *agent_role*, either Pacman or the ghosts are driven by
        the agent's action while the opponent uses its built-in AI (or an
        optional external policy).
        """
        if self.agent_role == "pacman":
            pacman_action = int(action)
            ghost_actions = self._get_opponent_ghost_actions()
        else:
            pacman_action = self._get_opponent_pacman_action()
            ghost_actions = [int(a) for a in action]

        return pacman_action, ghost_actions

    def _get_opponent_ghost_actions(self) -> Optional[list[Optional[int]]]:
        """Get ghost actions from the opponent policy, or None for built-in AI."""
        if self.opponent_policy is not None:
            state = self._game.get_state()
            actions = self.opponent_policy(state)
            return [int(a) for a in actions]
        return None  # Built-in AI

    def _get_opponent_pacman_action(self) -> Optional[int]:
        """Get the Pacman action from the opponent policy, or None for built-in AI."""
        if self.opponent_policy is not None:
            state = self._game.get_state()
            return int(self.opponent_policy(state))
        return None  # Built-in AI

    def _build_info(self) -> dict:
        return {
            "score": self._game.score,
            "lives": self._game.pacman.lives,
            "game_over": self._game.game_over,
            "level_complete": self._game.level_complete,
            "steps": self._steps,
        }
