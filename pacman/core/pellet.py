import pygame
from typing import Tuple
from utils.constants import PELLET_COLOR, POWERUP_COLOR

class Pellet:
    def __init__(self, position: Tuple[int, int], tile_size: int = 32):
        self.position = position[0] * tile_size + tile_size // 2, position[1] * tile_size + tile_size // 2
        self.eaten = False
        self.radius = 2
        self.color = PELLET_COLOR

    def draw(self, screen: pygame.Surface, tile_size: int) -> None:
        if not self.eaten:
            x, y = self.position
            pygame.draw.circle(screen, self.color, (x, y), self.radius)

    def check_collision(self, pacman_position: Tuple[int, int], pacman_radius: int) -> bool:
        x1, y1 = self.position
        x2, y2 = pacman_position
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance <= (self.radius + pacman_radius)

class PowerPellet(Pellet):
    def __init__(self, position: Tuple[int, int], tile_size: int = 32):
        super().__init__(position, tile_size)
        self.radius = 4
        self.color = POWERUP_COLOR
