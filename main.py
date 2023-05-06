import argparse
from training.train import train
from evaluation.evaluate import evaluate
from environments.pacman_environment import PacmanEnvironment
from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent
from models.pacman_model import PacmanModel
from models.ghost_model import GhostModel
import gym
import os
import torch

def load_models(model_dir):
    pacman_model = PacmanModel()
    ghost_model = GhostModel()

    pacman_model_path = os.path.join(model_dir, 'pacman_model.pth')
    ghost_model_paths = [os.path.join(model_dir, f'ghost_model_{i}.pth') for i in range(4)]

    if os.path.exists(pacman_model_path):
        pacman_model.load_state_dict(torch.load(pacman_model_path))
        print("Pacman model loaded.")

    for i, ghost_model_path in enumerate(ghost_model_paths):
        if os.path.exists(ghost_model_path):
            ghost_model.load_state_dict(torch.load(ghost_model_path))
            print(f"Ghost model {i} loaded.")

    return pacman_model, ghost_model

def main(args):
    # Create the Pacman environment
    env = PacmanEnvironment()

    # Load or create the Pacman and Ghost models
    pacman_model, ghost_model = load_models(args.model_dir)

    # Create the Pacman and Ghost agents
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    pacman_agent = PacmanAgent(state_size, action_size, model=pacman_model)
    ghost_agents = [GhostAgent(state_size, action_size, model=ghost_model) for _ in range(4)]

    if args.mode == 'train':
        train(args.num_episodes, args.model_dir)
    elif args.mode == 'evaluate':
        evaluate(env, pacman_agent, ghost_agents, args.num_episodes, args.model_dir)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'evaluate'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate'], help="Choose the mode: 'train' or 'evaluate'.")
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to run.')
    parser.add_argument('--save_interval', type=int, default=100, help='Interval for updating the models during training.')
    parser.add_argument('--model_dir', default='models', help='Directory to save/load the models.')
    args = parser.parse_args()
    main(args)