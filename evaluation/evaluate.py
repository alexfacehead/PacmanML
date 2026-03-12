"""Evaluate trained Pacman and/or Ghost DQN agents."""

import os
import random
import time
from typing import Optional

import numpy as np

from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent
from pacman.core.game import Game


def _resolve_level_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "pacman", "levels", "level_1.txt",
    )


def evaluate(
    checkpoint_dir: str = "checkpoints",
    num_episodes: int = 10,
    render: bool = True,
    agent_role: str = "both",
) -> None:
    """Evaluate trained agents.

    Args:
        checkpoint_dir: Directory containing pacman_agent.pt / ghost_agent.pt.
        num_episodes: Number of evaluation episodes.
        render: If True, show the game visually.
        agent_role: Which side(s) use the trained AI:
            - ``"both"``: both agents are AI (adversarial evaluation).
            - ``"pacman"``: only Pacman is AI; ghosts use built-in AI.
            - ``"ghosts"``: only ghosts are AI; Pacman uses random actions.
    """
    level_path = _resolve_level_path()
    game = Game(headless=(not render))
    game.load_level(level_path)

    # ---- Load agents ---------------------------------------------------
    pacman_agent: Optional[PacmanAgent] = None
    ghost_agent: Optional[GhostAgent] = None

    if agent_role in ("both", "pacman"):
        pacman_ckpt = os.path.join(checkpoint_dir, "pacman_agent.pt")
        if os.path.exists(pacman_ckpt):
            pacman_agent = PacmanAgent(device="auto")
            pacman_agent.load(pacman_ckpt)
            pacman_agent.epsilon = 0.0  # greedy at eval time
            print(f"Loaded Pacman agent from {pacman_ckpt}")
        else:
            print(f"WARNING: {pacman_ckpt} not found — Pacman will act randomly.")

    if agent_role in ("both", "ghosts"):
        ghost_ckpt = os.path.join(checkpoint_dir, "ghost_agent.pt")
        if os.path.exists(ghost_ckpt):
            ghost_agent = GhostAgent(device="auto")
            ghost_agent.load(ghost_ckpt)
            ghost_agent.epsilon = 0.0
            print(f"Loaded Ghost agent from {ghost_ckpt}")
        else:
            print(f"WARNING: {ghost_ckpt} not found — ghosts will use built-in AI.")

    # ---- Rendering setup -----------------------------------------------
    clock = None
    if render:
        import pygame
        pygame.init()
        from pacman.utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("PacmanML — Evaluation")
        clock = pygame.time.Clock()
        game.renderer.init_display(screen)
        game._screen = screen
        game._clock = clock

    # ---- Evaluation loop -----------------------------------------------
    max_steps = 5000
    scores: list[int] = []
    pac_rewards: list[float] = []
    survival_steps: list[int] = []
    wins: list[bool] = []

    for ep in range(num_episodes):
        game.reset()
        ep_reward = 0.0
        ep_steps = 0
        state = game.get_state()

        while not game.is_done() and ep_steps < max_steps:
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        _print_stats(scores, pac_rewards, survival_steps, wins)
                        return
                game.renderer.draw(
                    game.level, game.pacman, game.ghosts,
                    game.score, game.pacman.lives,
                    game_over=game.game_over,
                    level_complete=game.level_complete,
                    ready=(game.ready_timer > 0),
                    mode=game.current_mode,
                    paused=False,
                    score_popups=game._score_popups,
                )
                clock.tick(FPS)

            # Pacman action
            if pacman_agent is not None:
                pac_action = pacman_agent.select_action(state)
            else:
                pac_action = random.randint(0, 4)

            # Ghost actions (None = built-in AI)
            if ghost_agent is not None:
                g_actions = ghost_agent.select_action(state)
            else:
                g_actions = None

            game.step(pacman_action=pac_action, ghost_actions=g_actions)
            state = game.get_state()
            ep_reward += game.get_reward_pacman()
            ep_steps += 1

        scores.append(game.score)
        pac_rewards.append(ep_reward)
        survival_steps.append(ep_steps)
        wins.append(game.level_complete)

        outcome = "WIN" if game.level_complete else "LOSS"
        print(f"  Episode {ep + 1}/{num_episodes}: "
              f"Score={game.score:>5d}  Reward={ep_reward:>7.1f}  "
              f"Steps={ep_steps:>4d}  {outcome}")

    if render:
        import pygame
        pygame.quit()

    _print_stats(scores, pac_rewards, survival_steps, wins)


def _print_stats(
    scores: list[int],
    pac_rewards: list[float],
    survival_steps: list[int],
    wins: list[bool],
) -> None:
    """Print summary statistics."""
    if not scores:
        print("No episodes completed.")
        return
    print("\n--- Evaluation Summary ---")
    print(f"  Episodes        : {len(scores)}")
    print(f"  Avg score       : {np.mean(scores):.1f}")
    print(f"  Max score       : {np.max(scores)}")
    print(f"  Avg reward      : {np.mean(pac_rewards):.1f}")
    print(f"  Avg survival    : {np.mean(survival_steps):.0f} steps")
    print(f"  Win rate        : {np.mean(wins) * 100:.1f}%")
    print("--------------------------")


if __name__ == "__main__":
    evaluate()
