import pygame
from typing import Tuple
from core.level import Level
from core.pacman import Pacman
from random import choice

class Ghost:
    def __init__(self, start_position: Tuple[int, int]):
        self.position = start_position
        self.direction = None
        self.target = None

    def update(self, level: Level, pacman: Pacman) -> None:
        self.update_ai_state()
        next_position = self.calculate_next_position(level, pacman)
        self.position = next_position

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, (255, 0, 0), (self.position[0] * 32 + 16, self.position[1] * 32 + 16), 16)

    def update_ai_state(self) -> None:
        # Update the ghost's AI state based on timers and game events
        # This is a simplified example and can be expanded to include more complex AI logic
        pass

    def calculate_next_position(self, level: Level, pacman: Pacman) -> Tuple[int, int]:
        possible_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        valid_moves = []
        for direction in possible_directions:
            new_x = self.position[0] + direction[0]
            new_y = self.position[1] + direction[1]
            if not level.is_wall(new_x, new_y):
                valid_moves.append((new_x, new_y))

        if valid_moves:
            return choice(valid_moves)
        else:
            return self.position