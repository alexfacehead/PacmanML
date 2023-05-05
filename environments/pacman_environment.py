import gym
from typing import Tuple, List
from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent

class PacmanEnvironment(gym.Env):
    def __init__(self):
        # Initialize the environment
        pass

    def reset(self) -> Tuple:
        # Reset the environment and return the initial state
        pass

    def step(self, pacman_action: int, ghost_actions: List[int]) -> Tuple:
        # Execute actions for Pacman and Ghosts, and return the new state, reward, done, and info
        pass

    def render(self, mode: str = 'human') -> None:
        # Render the environment
        pass

    def close(self) -> None:
        # Close the environment
        pass

    def seed(self, seed: int = None) -> None:
        # Set the random seed for the environment
        pass

    def get_legal_actions(self, agent: PacmanAgent or GhostAgent) -> List[int]:
        # Get the legal actions for the given agent
        pass

    def is_terminal(self) -> bool:
        # Check if the current state is terminal (game over)
        pass