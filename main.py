"""PacmanML -- Adversarial Machine Learning for Pacman.

Usage:
    python main.py play [--play-as pacman|ghost] [--test] [--vs-ai]
    python main.py train [--episodes N] [--render] [--resume]
    python main.py watch [--checkpoint-dir DIR] [--episodes N] [--speed N]
    python main.py evaluate [--episodes N] [--render] [--agent-role both|pacman|ghosts]
"""

import argparse
import os
import sys


def _level_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pacman", "levels", "level_1.txt",
    )


def cmd_play(args: argparse.Namespace) -> None:
    """Launch the game for human play, optionally against trained AI."""
    from pacman.core.game import Game

    game = Game(headless=False, play_as=args.play_as)
    game.load_level(_level_path())

    if args.test:
        game.enable_test_mode(num_pellets=10)

    if not args.vs_ai:
        game.run()
        return

    # --- Play against trained AI ---
    import pygame
    from pacman.utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS

    ckpt_dir = args.checkpoint_dir

    if args.play_as == "ghost":
        # Human is ghost, load ML Pacman
        from agents.pacman_agent import PacmanAgent
        pac_ckpt = os.path.join(ckpt_dir, "pacman_agent.pt")
        if not os.path.exists(pac_ckpt):
            print(f"ERROR: No Pacman checkpoint at {pac_ckpt}")
            print("Train agents first:  python main.py train --episodes 1000")
            return
        pacman_ai = PacmanAgent(device="auto")
        pacman_ai.load(pac_ckpt)
        pacman_ai.epsilon = 0.0
        ghost_ai = None
        print(f"Playing as GHOST vs ML Pacman (loaded from {pac_ckpt})")
    else:
        # Human is Pacman, load ML Ghosts
        from agents.ghost_agent import GhostAgent
        ghost_ckpt = os.path.join(ckpt_dir, "ghost_agent.pt")
        if not os.path.exists(ghost_ckpt):
            print(f"ERROR: No Ghost checkpoint at {ghost_ckpt}")
            print("Train agents first:  python main.py train --episodes 1000")
            return
        ghost_ai = GhostAgent(device="auto")
        ghost_ai.load(ghost_ckpt)
        ghost_ai.epsilon = 0.0
        pacman_ai = None
        print(f"Playing as PACMAN vs ML Ghosts (loaded from {ghost_ckpt})")

    # Setup display
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    caption = "PacmanML — You vs AI"
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    game.renderer.init_display(screen)
    game._screen = screen
    game._clock = clock

    while game.running:
        game.handle_events()
        if game.paused:
            game.renderer.draw(
                game.level, game.pacman, game.ghosts,
                game.score, game.pacman.lives,
                game_over=game.game_over, level_complete=game.level_complete,
                ready=(game.ready_timer > 0), mode=game.current_mode,
                paused=True, score_popups=game._score_popups,
            )
            clock.tick(FPS)
            continue

        # AI actions from state
        state = game.get_state()
        pac_action = None
        g_actions = None
        if pacman_ai is not None:
            pac_action = pacman_ai.select_action(state)
        if ghost_ai is not None:
            g_actions = ghost_ai.select_action(state)

        game.step(pacman_action=pac_action, ghost_actions=g_actions)

        game.renderer.draw(
            game.level, game.pacman, game.ghosts,
            game.score, game.pacman.lives,
            game_over=game.game_over, level_complete=game.level_complete,
            ready=(game.ready_timer > 0), mode=game.current_mode,
            paused=False, score_popups=game._score_popups,
        )
        clock.tick(FPS)

    pygame.quit()


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch trained ML agents play against each other."""
    import pygame
    from pacman.core.game import Game
    from pacman.utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
    from agents.pacman_agent import PacmanAgent
    from agents.ghost_agent import GhostAgent

    ckpt_dir = args.checkpoint_dir

    # Load agents
    pac_ckpt = os.path.join(ckpt_dir, "pacman_agent.pt")
    ghost_ckpt = os.path.join(ckpt_dir, "ghost_agent.pt")

    pacman_ai = None
    ghost_ai = None

    if os.path.exists(pac_ckpt):
        pacman_ai = PacmanAgent(device="auto")
        pacman_ai.load(pac_ckpt)
        pacman_ai.epsilon = 0.0
        print(f"Loaded Pacman agent from {pac_ckpt}")
    else:
        print(f"WARNING: {pac_ckpt} not found — Pacman will act randomly")

    if os.path.exists(ghost_ckpt):
        ghost_ai = GhostAgent(device="auto")
        ghost_ai.load(ghost_ckpt)
        ghost_ai.epsilon = 0.0
        print(f"Loaded Ghost agent from {ghost_ckpt}")
    else:
        print(f"WARNING: {ghost_ckpt} not found — Ghosts will use built-in AI")

    # Setup game + display
    game = Game(headless=False)
    game.load_level(_level_path())

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PacmanML — AI vs AI")
    clock = pygame.time.Clock()
    game.renderer.init_display(screen)
    game._screen = screen
    game._clock = clock

    import random
    target_fps = FPS * args.speed
    episode = 0
    total_episodes = args.episodes

    while game.running and episode < total_episodes:
        game.reset()
        ep_steps = 0
        print(f"\n  Episode {episode + 1}/{total_episodes}")

        while game.running and not game.is_done() and ep_steps < 5000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game.running = False

            state = game.get_state()
            pac_action = pacman_ai.select_action(state) if pacman_ai else random.randint(0, 4)
            g_actions = ghost_ai.select_action(state) if ghost_ai else None

            game.step(pacman_action=pac_action, ghost_actions=g_actions)

            game.renderer.draw(
                game.level, game.pacman, game.ghosts,
                game.score, game.pacman.lives,
                game_over=game.game_over, level_complete=game.level_complete,
                ready=(game.ready_timer > 0), mode=game.current_mode,
                paused=False, score_popups=game._score_popups,
            )
            clock.tick(target_fps)
            ep_steps += 1

        result = "PAC WIN" if game.level_complete else "GHOST WIN" if game.game_over else "TIMEOUT"
        print(f"  Score: {game.score}  |  {result}  |  Steps: {ep_steps}")
        episode += 1

        # Brief pause between episodes
        if game.running and episode < total_episodes:
            pygame.time.wait(1000)

    pygame.quit()
    print(f"\nDone — watched {episode} episode(s).")


def cmd_train(args: argparse.Namespace) -> None:
    """Run adversarial DQN training."""
    from training.train import train

    train(
        config_path=args.config,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        render=args.render,
        resume=args.resume,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained agents."""
    from evaluation.evaluate import evaluate

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.episodes,
        render=args.render,
        agent_role=args.agent_role,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PacmanML -- Adversarial Machine Learning for Pacman",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- play ----------------------------------------------------------
    sp_play = subparsers.add_parser("play", help="Play the game interactively")
    sp_play.add_argument(
        "--play-as", choices=["pacman", "ghost"], default="pacman",
        help="Control Pacman or the lead ghost (default: pacman)",
    )
    sp_play.add_argument(
        "--test", action="store_true",
        help="Test mode: only 10 pellets near Pacman spawn",
    )
    sp_play.add_argument(
        "--vs-ai", action="store_true",
        help="Play against a trained AI opponent",
    )
    sp_play.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Checkpoint directory for --vs-ai (default: checkpoints)",
    )

    # ---- watch ---------------------------------------------------------
    sp_watch = subparsers.add_parser("watch", help="Watch trained AI agents play")
    sp_watch.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    sp_watch.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to watch (default: 5)",
    )
    sp_watch.add_argument(
        "--speed", type=int, default=1,
        help="Playback speed multiplier (default: 1)",
    )

    # ---- train ---------------------------------------------------------
    sp_train = subparsers.add_parser("train", help="Train agents adversarially")
    sp_train.add_argument(
        "--episodes", type=int, default=5000,
        help="Number of training episodes (default: 5000)",
    )
    sp_train.add_argument(
        "--render", action="store_true",
        help="Show the game during training (slower)",
    )
    sp_train.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    sp_train.add_argument(
        "--save-dir", default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)",
    )
    sp_train.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: config/config.yml)",
    )

    # ---- evaluate ------------------------------------------------------
    sp_eval = subparsers.add_parser("evaluate", help="Evaluate trained agents")
    sp_eval.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    sp_eval.add_argument(
        "--render", action="store_true",
        help="Show the game during evaluation",
    )
    sp_eval.add_argument(
        "--agent-role", choices=["both", "pacman", "ghosts"], default="both",
        help="Which side uses the trained AI (default: both)",
    )
    sp_eval.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory containing saved agent checkpoints",
    )

    args = parser.parse_args()

    if args.command == "play":
        cmd_play(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
