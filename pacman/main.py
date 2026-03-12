"""Entry point for the PacmanML game.

Run from project root:
    python -m pacman.main [--headless] [--play-as pacman|ghost]
"""

import argparse
import os

from .core.game import Game


def main():
    parser = argparse.ArgumentParser(description="PacmanML")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (for training)")
    parser.add_argument("--play-as", choices=["pacman", "ghost"], default="pacman",
                        help="Control pacman or ghost[0] (default: pacman)")
    parser.add_argument("--level", default=None,
                        help="Path to level file (default: levels/level_1.txt)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only 10 pellets near Pacman spawn")
    args = parser.parse_args()

    level_file = args.level
    if level_file is None:
        level_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "levels", "level_1.txt",
        )

    game = Game(headless=args.headless, play_as=args.play_as)
    game.load_level(level_file)

    if args.test:
        game.enable_test_mode(num_pellets=10)

    if args.headless:
        # Headless demo: run 1000 steps with random actions
        import random
        for _ in range(1000):
            game.step(pacman_action=random.randint(0, 3))
            if game.is_done():
                break
        print(f"Score: {game.score}, Game Over: {game.game_over}, "
              f"Level Complete: {game.level_complete}")
    else:
        game.run()


if __name__ == "__main__":
    main()
