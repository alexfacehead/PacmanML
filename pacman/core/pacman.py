import pygame
from typing import Tuple
from core.level import Level
from core.ghost import Ghost
from core.pellet import Pellet
from utils.constants import *

class Pacman:
    def __init__(self, start_position: Tuple[int, int]):
        self.start_position = start_position
        self.position = start_position
        self.direction = None
        self.next_direction = None
        self.powered_up = False
        self.power_up_timer = 0
        self.lives = PACMAN_LIVES
        self.radius = 16

    def handle_key_event(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_UP:
            self.next_direction = (0, -1)
        elif event.key == pygame.K_DOWN:
            self.next_direction = (0, 1)
        elif event.key == pygame.K_LEFT:
            self.next_direction = (-1, 0)
        elif event.key == pygame.K_RIGHT:
            self.next_direction = (1, 0)

    def update(self, level: Level) -> None:
        if self.next_direction:
            next_position = self.get_next_position(self.next_direction)
            if not level.is_wall(*next_position):
                self.direction = self.next_direction

        if self.direction:
            next_position = self.get_next_position(self.direction)
            if not level.is_wall(*next_position):
                self.position = next_position

        self.update_power_up_state()

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, (255, 255, 0), (self.position[0] * 32 + 16, self.position[1] * 32 + 16), 16)

    def check_collision(self, ghost: Ghost) -> bool:
        return self.position == ghost.position

    def get_next_position(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        x, y = self.position
        dx, dy = direction
        return x + dx, y + dy

    def eat_pellet(self, pellet: Pellet) -> None:
        if pellet.is_power_up:
            self.powered_up = True
            self.power_up_timer = 200  # Adjust the timer based on your game's requirements

    def update_power_up_state(self) -> None:
        if self.powered_up:
            self.power_up_timer -= 1
            if self.power_up_timer <= 0:
                self.powered_up = False
                self.power_up_timer = 0

    def reset_position(self) -> None:
        self.position = self.start_position
        self.direction = None
        self.next_direction = None
    
    def lose_life(self) -> None:
        self.lives -= 1