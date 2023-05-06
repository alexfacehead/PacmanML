import torch
import numpy as np
from environments.pacman_environment import PacmanEnvironment
from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Initialize the environment and agents
env = PacmanEnvironment()
pacman_agent = PacmanAgent(state_size=(84, 84), action_size=env.action_space.n, device="cuda" if torch.cuda.is_available() else "cpu")
ghost_agents = [GhostAgent() for _ in range(4)]

# Set training parameters
num_episodes = 1000
update_target_every = 100

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Pacman chooses an action
        pacman_action = pacman_agent.choose_action(state)

        # Ghosts choose their actions
        ghost_actions = [ghost_agent.choose_action(state) for ghost_agent in ghost_agents]

        # Perform the actions in the environment
        next_state, reward, done, _ = env.step(pacman_action, ghost_actions)

        # Pacman learns from the experience
        pacman_agent.learn(state, pacman_action, reward, next_state, done)

        # Ghosts learn from the experience
        for ghost_agent in ghost_agents:
            ghost_agent.learn(state, pacman_action, reward, next_state, done)

        state = next_state
        episode_reward += reward

    # Update the target model for Pacman
    if episode % update_target_every == 0:
        pacman_agent.update_target_model()

    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()