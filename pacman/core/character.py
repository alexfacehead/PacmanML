"""Base character class for Pacman and ghosts."""

from typing import Tuple


class Character:
    """Base class for all moving characters on the grid."""

    def __init__(self, start_position: Tuple[int, int], speed: int):
        """
        Args:
            start_position: (x, y) grid coordinates.
            speed: Number of frames between moves (higher = slower).
        """
        self.start_position = start_position
        self.position = start_position
        self.direction: Tuple[int, int] = (0, 0)
        self.speed = speed
        self.move_timer = 0

    def can_move(self) -> bool:
        """Returns True when enough frames have passed to allow a move.
        Automatically increments and resets the internal timer."""
        self.move_timer += 1
        if self.move_timer >= self.speed:
            self.move_timer = 0
            return True
        return False

    def get_next_position(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        """Returns the grid position one step in the given direction."""
        return (self.position[0] + direction[0],
                self.position[1] + direction[1])

    def reset_position(self) -> None:
        """Reset to starting position and clear direction."""
        self.position = self.start_position
        self.direction = (0, 0)
        self.move_timer = 0

    def check_collision(self, other: "Character") -> bool:
        """True if this character occupies the same grid cell as *other*."""
        return self.position == other.position
