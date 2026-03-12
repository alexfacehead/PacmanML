"""Pacman character class."""

from typing import Tuple, Optional

from ..core.character import Character
from ..utils.constants import (
    PACMAN_SPEED, PACMAN_LIVES, POWERUP_DURATION,
    UP, DOWN, LEFT, RIGHT, STAY, ACTION_TO_DIR,
)


class Pacman(Character):
    """The player-controlled (or AI-controlled) Pacman."""

    def __init__(self, start_position: Tuple[int, int]):
        super().__init__(start_position, PACMAN_SPEED)
        self.next_direction: Tuple[int, int] = (0, 0)
        self.lives = PACMAN_LIVES
        self.powered_up = False
        self.power_up_timer = 0
        self.powerup_duration = POWERUP_DURATION  # can be overridden for headless
        self.score = 0
        # Animation state
        self.mouth_angle = 0
        self.mouth_opening = True
        self.frame_counter = 0

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def handle_key_event(self, key: int) -> None:
        """Buffer a direction from a pygame key constant."""
        import pygame
        if key == pygame.K_UP:
            self.next_direction = (0, -1)
        elif key == pygame.K_DOWN:
            self.next_direction = (0, 1)
        elif key == pygame.K_LEFT:
            self.next_direction = (-1, 0)
        elif key == pygame.K_RIGHT:
            self.next_direction = (1, 0)

    def set_action(self, action: int) -> None:
        """Set direction from an ML action index (UP=0 DOWN=1 LEFT=2 RIGHT=3 STAY=4)."""
        if action in ACTION_TO_DIR:
            self.next_direction = ACTION_TO_DIR[action]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, level, headless: bool = False) -> None:
        """Move Pacman on the grid, respecting walls and wrapping."""
        # Try the buffered direction first
        if self.next_direction != (0, 0):
            nx, ny = self.get_next_position(self.next_direction)
            nx, ny = level.wrap_position(nx, ny)
            if not level.is_wall(nx, ny) and not level.is_ghost_house_door(nx, ny):
                self.direction = self.next_direction

        # Move in current direction if timer allows
        if self.can_move() and self.direction != (0, 0):
            nx, ny = self.get_next_position(self.direction)
            nx, ny = level.wrap_position(nx, ny)
            if not level.is_wall(nx, ny) and not level.is_ghost_house_door(nx, ny):
                self.position = (nx, ny)

        # Update power-up timer
        if self.powered_up:
            self.power_up_timer -= 1
            if self.power_up_timer <= 0:
                self.powered_up = False
                self.power_up_timer = 0

        # Mouth animation (skip in headless — purely visual)
        if not headless:
            self._update_animation()

    def _update_animation(self) -> None:
        """Update the mouth chomping animation."""
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            if self.mouth_opening:
                self.mouth_angle += 5
                if self.mouth_angle >= 45:
                    self.mouth_opening = False
            else:
                self.mouth_angle -= 5
                if self.mouth_angle <= 5:
                    self.mouth_opening = True

    def activate_power_up(self) -> None:
        """Activate the power pellet effect."""
        self.powered_up = True
        self.power_up_timer = self.powerup_duration

    def lose_life(self) -> None:
        """Decrement lives."""
        self.lives -= 1

    def reset(self) -> None:
        """Full reset for a new game (position, lives, score, power-up)."""
        self.reset_position()
        self.next_direction = (0, 0)
        self.lives = PACMAN_LIVES
        self.powered_up = False
        self.power_up_timer = 0
        self.score = 0

    def reset_position(self) -> None:
        """Reset position and direction only (for life loss)."""
        super().reset_position()
        self.next_direction = (0, 0)
