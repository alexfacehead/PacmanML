import gym
import gym_pacman
import numpy as np
from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent
from typing import Tuple

def train(num_episodes: int, model_save_path: str = "models/"):
    env = gym.make("Pacman-v0")
    state_size = (4, 84, 84)
    action_size = env.action_space.n
    num_ghosts = 4

    pacman_agent = PacmanAgent(state_size, action_size)
    ghost_agents = [GhostAgent(state_size, action_size) for _ in range(num_ghosts)]

    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        done = False

        while not done:
            pacman_action = pacman_agent.choose_action(state)
            ghost_actions = [ghost_agent.choose_action(state) for ghost_agent in ghost_agents]

            next_state, rewards, done, _ = env.step(pacman_action, ghost_actions)
            next_state = preprocess_state(next_state)

            pacman_reward, ghost_rewards = rewards
            pacman_agent.learn(state, pacman_action, pacman_reward, next_state, done)
            for i, ghost_agent in enumerate(ghost_agents):
                ghost_agent.learn(state, ghost_actions[i], ghost_rewards[i], next_state, done)

            state = next_state

        if episode % 10 == 0:
            pacman_agent.update_target_model()
            for ghost_agent in ghost_agents:
                ghost_agent.update_target_model()

    pacman_agent.model.save(model_save_path + "pacman_model.pth")
    for i, ghost_agent in enumerate(ghost_agents):
        ghost_agent.model.save(model_save_path + f"ghost_model_{i}.pth")

def preprocess_state(state: np.ndarray) -> np.ndarray:
    # Preprocess the state as needed, e.g., grayscale, resize, normalize, etc.
    pass