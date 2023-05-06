import pygame

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 32

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
PINK = (255, 192, 203)
ORANGE = (255, 165, 0)
PELLET_COLOR = WHITE  # Add this line
POWERUP_COLOR = GREEN  # Add this line

# Pacman settings
PACMAN_SPEED = 2
POWERUP_DURATION = 10  # In seconds
PACMAN_LIVES = 3
PACMAN_COLOR = YELLOW
PELLET_SCORE = 10
POWERUP_SCORE = 50
GHOST_SCORE = 200

# Ghost settings
GHOST_SPEED = 1
SCATTER_MODE_DURATION = 7  # In seconds
CHASE_MODE_DURATION = 20  # In seconds
FRIGHTENED_MODE_DURATION = 10  # In seconds
BLINKY_COLOR = RED
PINKY_COLOR = PINK
INKY_COLOR = CYAN
CLYDE_COLOR = ORANGE

# Level settings
TILE_SIZE = 32
WALL_COLOR = BLUE

# Fonts
pygame.font.init()
FONT_SMALL = pygame.font.Font(None, 24)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_LARGE = pygame.font.Font(None, 48)

# Visual assets
RESOURCE_PATH = "/home/dev/pacman/pacmanML/pacman/assets"