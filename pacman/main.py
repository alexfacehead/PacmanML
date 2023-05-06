import pygame
from utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from core.game import Game
import argparse

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    print("Start game!")
    pygame.display.set_caption("Pacman")
    clock = pygame.time.Clock()

    print("Attempting to initialize game.")
    game = Game(screen, clock, False)
    print("Attempting to load level.")
    game.load_level("levels/level_1.txt")
    game.run()

    pygame.quit()

if __name__ == "__main__":
    main()
