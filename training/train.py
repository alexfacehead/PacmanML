"""Adversarial training: Pacman DQN vs Ghost-team DQN.

Both agents observe the same state and train against each other.
Pacman is rewarded for eating pellets and surviving; ghosts are rewarded
for catching Pacman.  This creates an arms-race that pushes both sides
toward stronger play.
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import yaml

from agents.pacman_agent import PacmanAgent
from agents.ghost_agent import GhostAgent
from pacman.core.game import Game


def _resolve_level_path() -> str:
    """Return the absolute path to level_1.txt."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "pacman", "levels", "level_1.txt",
    )


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load training hyper-parameters from a YAML config file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, "config", "config.yml",
        )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _get_device_str(cfg: dict) -> str:
    """Extract device string from config, defaulting to 'auto'."""
    return cfg.get("training", {}).get("device", "auto")


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Simple progress bar string."""
    filled = int(width * min(value / max_val, 1.0)) if max_val > 0 else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def train(
    config_path: Optional[str] = None,
    num_episodes: int = 5000,
    save_dir: str = "checkpoints",
    render: bool = False,
    resume: bool = False,
) -> None:
    """Train Pacman and Ghost agents adversarially."""

    # ---- Config --------------------------------------------------------
    cfg = _load_config(config_path)
    tcfg = cfg.get("training", {})

    lr             = tcfg.get("learning_rate", 0.0001)
    gamma          = tcfg.get("gamma", 0.99)
    eps_start      = tcfg.get("epsilon_start", 1.0)
    eps_end        = tcfg.get("epsilon_end", 0.05)
    eps_decay      = tcfg.get("epsilon_decay", 0.9995)
    buffer_size    = tcfg.get("replay_buffer_size", 100_000)
    batch_size     = tcfg.get("batch_size", 64)
    target_update  = tcfg.get("target_update_freq", 1000)
    save_interval  = tcfg.get("save_interval", 500)
    device_str     = _get_device_str(cfg)

    # ---- Game ----------------------------------------------------------
    level_path = _resolve_level_path()
    game = Game(headless=(not render))
    game.load_level(level_path)
    total_pellets = game.level.total_pellets()

    # ---- Agents --------------------------------------------------------
    agent_kwargs = dict(
        in_channels=6,
        lr=lr,
        gamma=gamma,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update,
        device=device_str,
    )

    pacman_agent = PacmanAgent(num_actions=5, **agent_kwargs)
    ghost_agent  = GhostAgent(num_ghosts=4, num_actions=5, **agent_kwargs)

    # ---- Resume --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    pacman_ckpt = os.path.join(save_dir, "pacman_agent.pt")
    ghost_ckpt  = os.path.join(save_dir, "ghost_agent.pt")
    start_episode = 0

    if resume:
        if os.path.exists(pacman_ckpt):
            pacman_agent.load(pacman_ckpt)
        if os.path.exists(ghost_ckpt):
            ghost_agent.load(ghost_ckpt)
        meta_path = os.path.join(save_dir, "meta.npy")
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            start_episode = meta.get("episode", 0)

    # ---- Banner --------------------------------------------------------
    print()
    print("=" * 65)
    print("  PacmanML  --  Adversarial Training")
    print("=" * 65)
    print(f"  Episodes     : {num_episodes} (starting from {start_episode})")
    print(f"  Device       : {pacman_agent.device}")
    print(f"  LR           : {lr}")
    print(f"  Gamma        : {gamma}")
    print(f"  Epsilon      : {eps_start} -> {eps_end} (decay {eps_decay})")
    print(f"  Batch size   : {batch_size}")
    print(f"  Buffer size  : {buffer_size:,}")
    print(f"  Target update: every {target_update} steps")
    print(f"  Save interval: every {save_interval} episodes")
    print(f"  Save dir     : {os.path.abspath(save_dir)}")
    print(f"  Render       : {render}")
    if resume and start_episode > 0:
        print(f"  Resumed from : episode {start_episode}")
        print(f"  Pac epsilon  : {pacman_agent.epsilon:.4f}")
        print(f"  Ghost epsilon: {ghost_agent.epsilon:.4f}")
    print("=" * 65)
    print()
    print("  Ep   | Score |  Pellets | Result   | Pac R  | Gho R  | "
          "Eps    | Loss P  | Loss G  | Steps | Time")
    print("-" * 120)

    # ---- Rendering setup (if requested) --------------------------------
    if render:
        import pygame
        pygame.init()
        from pacman.utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("PacmanML — Training")
        clock = pygame.time.Clock()
        game.renderer.init_display(screen)
        game._screen = screen
        game._clock = clock

    # ---- Tracking ------------------------------------------------------
    max_steps_per_episode = 800  # each step is a real move in headless mode
    all_scores = []
    all_pac_rewards = []
    all_ghost_rewards = []
    recent_losses_pac = []
    recent_losses_ghost = []
    pac_wins = 0
    ghost_wins = 0
    timeouts = 0
    best_score = 0

    t0 = time.time()

    # ---- Training loop -------------------------------------------------
    for episode in range(start_episode, start_episode + num_episodes):
        game.reset()
        ep_pac_reward = 0.0
        ep_ghost_reward = 0.0
        ep_steps = 0

        state = game.get_state()

        while not game.is_done() and ep_steps < max_steps_per_episode:
            # Render (optional)
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("\n  Training interrupted by user.")
                        _save_all(pacman_agent, ghost_agent, pacman_ckpt,
                                  ghost_ckpt, save_dir, episode)
                        _print_summary(all_scores, pac_wins, ghost_wins,
                                       timeouts, best_score, t0)
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
                clock.tick(120)

            # Select actions
            pacman_action = pacman_agent.select_action(state)
            ghost_actions = ghost_agent.select_action(state)

            # Step game
            game.step(pacman_action=pacman_action, ghost_actions=ghost_actions)

            next_state = game.get_state()
            pac_reward = game.get_reward_pacman()
            ghost_reward = game.get_reward_ghost()
            done = game.is_done()

            # Store transitions
            pacman_agent.store_transition(state, pacman_action, pac_reward,
                                          next_state, done)
            ghost_agent.store_transition(state, ghost_actions, ghost_reward,
                                         next_state, done)

            # Train
            loss_p = pacman_agent.train_step()
            loss_g = ghost_agent.train_step()

            if loss_p is not None:
                recent_losses_pac.append(loss_p)
            if loss_g is not None:
                recent_losses_ghost.append(loss_g)

            ep_pac_reward += pac_reward
            ep_ghost_reward += ghost_reward
            state = next_state
            ep_steps += 1

        # ---- Episode results -------------------------------------------
        pellets_eaten = total_pellets - game.level.total_pellets()
        all_scores.append(game.score)
        all_pac_rewards.append(ep_pac_reward)
        all_ghost_rewards.append(ep_ghost_reward)

        if game.score > best_score:
            best_score = game.score

        if game.level_complete:
            result = "PAC WIN"
            pac_wins += 1
        elif game.game_over:
            result = "GHOST W"
            ghost_wins += 1
        else:
            result = "TIMEOUT"
            timeouts += 1

        # Print every episode
        ep_num = episode + 1
        elapsed = time.time() - t0
        eps_done = ep_num - start_episode
        eta = (elapsed / eps_done) * (num_episodes - eps_done) if eps_done > 0 else 0
        avg_lp = np.mean(recent_losses_pac[-100:]) if recent_losses_pac else 0.0
        avg_lg = np.mean(recent_losses_ghost[-100:]) if recent_losses_ghost else 0.0

        print(
            f"  {ep_num:>4d} | "
            f"{game.score:>5d} | "
            f"{pellets_eaten:>3d}/{total_pellets:<3d} | "
            f"{result:<8s} | "
            f"{ep_pac_reward:>+6.1f} | "
            f"{ep_ghost_reward:>+6.1f} | "
            f"{pacman_agent.epsilon:.4f} | "
            f"{avg_lp:>7.4f} | "
            f"{avg_lg:>7.4f} | "
            f"{ep_steps:>5d} | "
            f"{_fmt_time(elapsed)} (ETA {_fmt_time(eta)})"
        )

        # Summary line every 50 episodes
        if eps_done > 0 and eps_done % 50 == 0:
            last_50 = all_scores[-50:]
            last_50_pac = all_pac_rewards[-50:]
            print()
            print(f"  --- 50-ep avg: Score {np.mean(last_50):.0f} | "
                  f"Best {best_score} | "
                  f"Pac W {pac_wins} / Ghost W {ghost_wins} / TO {timeouts} | "
                  f"Avg Pac R {np.mean(last_50_pac):.1f} | "
                  f"Buffer {len(pacman_agent.replay_buffer):,} ---")
            print()

        # Save checkpoints periodically
        if eps_done % save_interval == 0:
            _save_all(pacman_agent, ghost_agent, pacman_ckpt, ghost_ckpt,
                      save_dir, ep_num)

    # ---- Final save & summary ------------------------------------------
    _save_all(pacman_agent, ghost_agent, pacman_ckpt, ghost_ckpt,
              save_dir, start_episode + num_episodes)
    _print_summary(all_scores, pac_wins, ghost_wins, timeouts, best_score, t0)

    if render:
        import pygame
        pygame.quit()


def _save_all(
    pacman_agent: PacmanAgent,
    ghost_agent: GhostAgent,
    pacman_ckpt: str,
    ghost_ckpt: str,
    save_dir: str,
    episode: int,
) -> None:
    """Persist both agents and a small metadata file."""
    pacman_agent.save(pacman_ckpt)
    ghost_agent.save(ghost_ckpt)
    np.save(os.path.join(save_dir, "meta.npy"), {"episode": episode})
    print(f"  >>> Checkpoint saved at episode {episode}")


def _print_summary(
    all_scores: list,
    pac_wins: int,
    ghost_wins: int,
    timeouts: int,
    best_score: int,
    t0: float,
) -> None:
    """Print a final training summary."""
    total = len(all_scores)
    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    if total > 0:
        print(f"  Total episodes : {total}")
        print(f"  Total time     : {_fmt_time(elapsed)}")
        print(f"  Avg score      : {np.mean(all_scores):.0f}")
        print(f"  Best score     : {best_score}")
        last_n = min(100, total)
        print(f"  Last {last_n} avg   : {np.mean(all_scores[-last_n:]):.0f}")
        print(f"  Pacman wins    : {pac_wins} ({100 * pac_wins / total:.1f}%)")
        print(f"  Ghost wins     : {ghost_wins} ({100 * ghost_wins / total:.1f}%)")
        print(f"  Timeouts       : {timeouts} ({100 * timeouts / total:.1f}%)")
    print("=" * 65)
    print()


if __name__ == "__main__":
    train()
