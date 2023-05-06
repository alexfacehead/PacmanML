import pygame
from typing import Tuple
from utils.constants import PELLET_COLOR, POWERUP_COLOR

class Pellet:
    def __init__(self, position: Tuple[int, int], is_power_pellet: bool = False, tile_size: int = 32):
        self.position = position[0] * tile_size + tile_size // 2, position[1] * tile_size + tile_size // 2
        self.is_power_pellet = is_power_pellet
        self.eaten = False
        self.radius = 4 if is_power_pellet else 2
        self.color = POWERUP_COLOR if is_power_pellet else PELLET_COLOR

    def draw(self, screen: pygame.Surface, tile_size: int) -> None:
        if not self.eaten:
            x, y = self.position
            pygame.draw.circle(screen, self.color, (x, y), self.radius)

    def check_collision(self, pacman_position: Tuple[int, int], pacman_radius: int) -> bool:
        x1, y1 = self.position
        x2, y2 = pacman_position
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance <= (self.radius + pacman_radius)