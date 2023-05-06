import pygame
from typing import Tuple
from core.level import Level
from core.pellet import Pellet, PowerPellet
from utils.constants import *
from core.character import Character

class Pacman(Character):
    def __init__(self, start_position: Tuple[int, int]):
        super().__init__(start_position, PACMAN_SPEED)  # Pass PACMAN_SPEED to the Character class constructor
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
        self.check_pellet_collision(level)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, (255, 255, 0), (self.position[0] * 32 + 16, self.position[1] * 32 + 16), 16)

    def check_collision(self, obj) -> bool:
        return self.position == obj.position

    def get_next_position(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        x, y = self.position
        dx, dy = direction
        return x + dx, y + dy

    def eat_pellet(self, level: Level, pellet: Pellet) -> None:
        if isinstance(pellet, PowerPellet):
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

    def check_pellet_collision(self, level: Level) -> None:
        for pellet in level.pellets:
            if self.check_collision(pellet):
                self.handle_pellet_collision(level, pellet)

        for power_pellet in level.power_pellets:
            if self.check_collision(power_pellet):
                self.handle_power_pellet_collision(level, power_pellet)

    def handle_pellet_collision(self, level: Level, pellet: Pellet) -> None:
        level.pellets.remove(pellet)
        self.eat_pellet(level, pellet)

    def handle_power_pellet_collision(self, level: Level, power_pellet: PowerPellet) -> None:
        level.power_pellets.remove(power_pellet)
        self.eat_pellet(level, power_pellet)
