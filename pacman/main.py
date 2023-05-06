import pygame
from utils.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from core.game import Game
from core.level import Level

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pacman")
    clock = pygame.time.Clock()

    game = Game(screen, clock)
    game.load_level("path/to/level_file.txt")
    game.run()

    pygame.quit()

if __name__ == "__main__":
    main()
