# Game constants for PacmanML

# Grid and screen
GRID_SIZE = 24
COLS = 28
ROWS = 31
SCREEN_WIDTH = COLS * GRID_SIZE   # 672
SCREEN_HEIGHT = ROWS * GRID_SIZE + GRID_SIZE  # 768 (31*24 + 24 for HUD)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (33, 33, 255)
CYAN = (0, 255, 255)
PINK = (255, 184, 255)
ORANGE = (255, 184, 82)
GREEN = (0, 255, 0)
DARK_BLUE = (0, 0, 40)

# Character speeds (frames between moves)
PACMAN_SPEED = 8
GHOST_SPEED = 9

# Timers (in frames at 60fps)
POWERUP_DURATION = 360       # 6 seconds -- matches frightened duration
FRIGHTENED_FLASH_TIME = 90   # start flashing with this many frames left

# Lives
PACMAN_LIVES = 3

# Scoring
PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_SCORE = 200

# Ghost mode durations (in frames)
SCATTER_DURATION = 420
CHASE_DURATION = 1200
FRIGHTENED_DURATION = 360    # 6 seconds -- same as POWERUP_DURATION

# Ghost colors
GHOST_COLORS = {
    "blinky": RED,
    "pinky": PINK,
    "inky": CYAN,
    "clyde": ORANGE,
}

# Ghost scatter targets (corners of the map)
GHOST_SCATTER_TARGETS = {
    "blinky": (25, 0),
    "pinky": (2, 0),
    "inky": (27, 30),
    "clyde": (0, 30),
}

# Ghost names in order
GHOST_NAMES = ["blinky", "pinky", "inky", "clyde"]

# Actions (for ML agents)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

ACTION_TO_DIR = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    STAY: (0, 0),
}

# Ghost states
SCATTER = 0
CHASE = 1
FRIGHTENED = 2
EATEN = 3

# Ghost house re-entry target (center of ghost house)
GHOST_HOUSE_TARGET = (13, 14)
GHOST_HOUSE_DOOR = (13, 12)

# FPS
FPS = 60

# Ready timer (frames to show READY! before game starts)
READY_TIMER = 120

# State channels for ML (indices into state array)
CHANNEL_WALLS = 0
CHANNEL_PELLETS = 1
CHANNEL_POWER_PELLETS = 2
CHANNEL_PACMAN = 3
CHANNEL_GHOSTS = 4
CHANNEL_FRIGHTENED = 5
NUM_STATE_CHANNELS = 6
