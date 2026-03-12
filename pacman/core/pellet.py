"""Pellet and PowerPellet classes -- simple data objects on the grid."""

from typing import Tuple


class Pellet:
    """A regular pellet at a grid position."""

    def __init__(self, position: Tuple[int, int]):
        self.position = position  # (x, y) in grid coords
        self.eaten = False
        self.is_power = False


class PowerPellet(Pellet):
    """A power pellet that lets Pacman eat ghosts."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        self.is_power = True
